import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from utils.bayes.media_transforms import adstock, shape

def load_gt_data(file_name:str) -> pd.DataFrame:
    data = pd.read_csv(file_name, skiprows=1, parse_dates=["Week"])
    data.columns = [col.replace(": (United States)", "").lower().replace(" ","_") for col in data.columns]
    return data

def add_time_trend(data:pd.DataFrame):
    data = data.copy()
    data["t"] = np.arange(1, len(data) + 1)
    return data

def merge_gt_data(data:pd.DataFrame, gt_data:pd.DataFrame) -> pd.DataFrame:
    data = data.merge(gt_data, on="Week", how="left")
    return data

def gt_data_to_impressions(data:pd.DataFrame, col: str, scale=10_000) -> pd.DataFrame:
    data = data.copy()
    data[col] = (data[col] + 5e-3) * scale * np.exp(0.01*np.random.randn(len(data))) # Add small value to avoid log(0)
    data[col] = data[col].astype(int)
    return data
  
def generate_season(time, period=52.18, terms=4):
  betas_sin = np.random.randn(terms)
  betas_cos = np.random.randn(terms)
  
  sines = jnp.sum([jnp.sin(2*jnp.pi*(i+1)*time/period)/(2*(i+1)**1.4) for i in range(terms)]*betas_sin[None, :].T, axis=0)
  cosines = jnp.sum([jnp.cos(2*jnp.pi*(i+1)*time/period)/(2*(i+1)**1.4) for i in range(terms)]*betas_cos[None, :].T, axis=0)
  season_unadjusted = sines + cosines
  season = (season_unadjusted - jnp.min(season_unadjusted))
  season = season/jnp.max(season)
  return season - .5

def make_dataset(
  data: pd.DataFrame, 
  imp_col:str, 
  click_col:str, 
  time_col:str, 
  n_media_channels=8,
  media_var=.4,
  media_mean=7,
  prob_media_on=.7,
  media_season_strength=.1):
  
  m_ = np.random.normal(0, .2, size=(n_media_channels,))[:, None]
  s_ = np.random.normal(0, .5, size=(n_media_channels,))[:, None]
  
  season = generate_season(data[time_col].values)[None, :]
  media_data = np.exp(
    np.random.normal(
    media_mean
    + m_*(data[time_col]/data[time_col].max()).values[None, :] 
    + media_season_strength*season
    , 
    media_var, 
    size=(n_media_channels, len(data))
    )
  ).astype(int).T * np.random.binomial(
    1, prob_media_on, 
    size=(len(data), n_media_channels)
    )

  media_betas_imps = np.abs(np.random.randn(n_media_channels)*.03+0.01)
  media_saturation_imps = (np.random.beta(1, 3, size=(n_media_channels,))*1 + 1.2)
  media_shape_imps = (np.random.beta(1, 3, size=(n_media_channels,))*3 + .9)
  
  
  media_betas_clicks = np.abs(np.random.randn(n_media_channels)*.01 + media_betas_imps)
  media_saturation_clicks = np.abs(np.random.randn(n_media_channels)*0.05 + media_saturation_imps)
  media_shape_clicks = np.abs(np.random.randn(n_media_channels)*0.005 + media_shape_imps)
  
  decay = np.random.beta(1, 3, size=(n_media_channels,))*0.2
  delay = np.floor(np.random.beta(3, 2, size=(n_media_channels,))*3)
  
  media_shape_transformed_imps = shape.hill_function(media_data/media_data.mean(axis=0), n=media_shape_imps, k=media_saturation_imps)
  media_shape_transformed_clicks = shape.hill_function(media_data/media_data.mean(axis=0), n=media_shape_clicks, k=media_saturation_clicks)
  
  media_transformed_imps = adstock.delayed_adstock(
    media_shape_transformed_imps, 
    alpha=decay, 
    theta=delay,
    normalize=True
  ).eval()
  media_transformed_clicks = adstock.delayed_adstock(
    media_shape_transformed_clicks,
    alpha=decay,
    theta=delay,
    normalize=True
  ).eval()
  
  media_effect_imps = np.sum(media_transformed_imps*media_betas_imps, axis=1)
  media_effect_clicks = np.sum(media_transformed_clicks*media_betas_clicks, axis=1)
  
  impressions = jax.random.poisson(jax.random.PRNGKey(0),
    data[imp_col].values 
    * np.exp(media_effect_imps) 
    * np.exp(0.1*season.T[:,0]),
    shape = (len(media_effect_imps),))
  
  jsp_logit = jax.scipy.special.logit
  logit_ctr = jsp_logit(
    np.clip(
      (data[click_col]/data[imp_col]).values, 
      0.1, 0.8)
    )
  ctr_logit = logit_ctr + 3*media_effect_clicks + .3*season.T[:,0]
  
  clicks = jax.random.binomial(
    jax.random.PRNGKey(0),
    impressions,
    jax.nn.sigmoid(ctr_logit),
    shape = (len(ctr_logit),))
  
  param_dict = {
    "media_betas_imps": media_betas_imps,
    "media_saturation_imps": media_saturation_imps,
    "media_shape_imps": media_shape_imps,
    "media_betas_clicks": media_betas_clicks,
    "media_saturation_clicks": media_saturation_clicks,
    "media_shape_clicks": media_shape_clicks,
    "decay": decay,
    "delay": delay
  }
  return media_data, impressions, clicks, ctr_logit, param_dict
  