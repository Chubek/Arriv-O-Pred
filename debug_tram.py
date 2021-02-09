import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import datetime
import json
import optparse

def calculate_sum_time(config_file, source_location, target_location, first_ata, reverse=False):   
  config = json.load(open(config_file))
  print(f"Config file {config_file} loaded!")
  xgb_model_prime = pickle.load(open(config["prime_model"], "rb"))
  xgb_model_second = pickle.load(open(config["second_model"], "rb"))

  dd = pd.read_csv(config["loc_dataset"])
  df = pd.read_csv(config["loc_predictions"])

  locations_be = dd.reset_index().loc[dd[dd["location"] == source_location].index.min():dd[dd["location"] == target_location].index.min(), ["location"]]
  locations_ne = dd.reset_index().loc[dd[dd["location"] == target_location].index.min():dd[dd["location"] == source_location].index.min(), ["location"]]

  if locations_ne.shape[0] > 1:
    locations = locations_ne[::-1]
  else:
    locations = locations_be

  print(f"Locations loaded. They are  {[l[0] for l in list(locations.values)]}")

  le = LabelEncoder()

  le.fit(dd.loc[:, ["location"]].values.ravel())
  
  locations_label = le.transform(locations.values.ravel())
  
  print(f"Locations converted to label. Labels are {locations_label}")

  atas = [first_ata]
  atds = []
  predicted_a = []
  predicted_b = []
  
  for i, loc in enumerate(locations_label):
    
    print(f"{i}-th location...")
    
    X_prime = pd.DataFrame.from_dict({"location": [loc], "actual_ta": [atas[-1]], "gbtt_ptd": [int(df[df["loc"] ==  le.inverse_transform([loc])[0]]["ptd"])]})

    X_prime_pred = int(xgb_model_prime.predict(X_prime)[0])

    print(f"Primary Prediction {X_prime_pred} made.")

    ata_ls = list(str(atas[-1]))

    if ata_ls[0:2] == ["2", "4"]:
      ata_ls[0:2] = ["0", "0"]
    
    ata_n = "".join(ata_ls)

    ata_dt = datetime.datetime.strptime(ata_n, "%H%M")

    dt_1 = ata_dt + datetime.timedelta(minutes=X_prime_pred)
    atds.append(int(dt_1.strftime("%H%M")))
    
    print(f"Departures appended with {atds[-1]}")

    X_second = pd.DataFrame.from_dict({"location": [loc], "actual_td": [atds[-1]], "gbtt_pta": [int(df[df["loc"] == le.inverse_transform([loc])[0]]["pta"])]})

    X_second_pred = int(xgb_model_second.predict(X_second)[0])
    
    print(f"Secondary prediction {X_second_pred} made")

    dt_2 = ata_dt + datetime.timedelta(minutes=X_second_pred)
    
    if dt_2.strftime("%H%M")[:2] == "00":
      ata_ls = list(dt_2.strftime("%H%M"))
      ata_ls[:2] = ["2", "4"]
      ata = int("".join(ata_ls))
    else:     
      ata = int(dt_2.strftime("%H%M"))

    atas.append(ata)

    print(f"Arrivals appended with {atas[-1]}")

    predicted_a.append(X_prime_pred), predicted_b.append(X_second_pred)
    
  atas = atas[:-1]
  locat_out = [l[0] for l in list(locations.values)]

  if reverse:
    print(f"Reversing...")
    atas.reverse()
    atds.reverse()
    predicted_a.reverse()
    predicted_b.reverse()
    locat_out.reverse()

  return {"locations":locat_out, "arrivals": atas, "departures": atds, "predicted_a": predicted_a, "predicted_b": predicted_b}



if __name__ == "__main__":
  parser = optparse.OptionParser()

  parser.add_option('-c', '--config',
    action="store", dest="config",
    help="Config file", default="config.json")

  parser.add_option('-s', '--source',
    action="store", dest="source",
    help="Source")

  parser.add_option('-d', '--dest',
    action="store", dest="dest",
    help="Destination")

  parser.add_option('-a', '--arrival',
    action="store", dest="arrival",
    help="Arrival time", type="int")

  parser.add_option('-i', '--inverse',
  const=True, dest="inverse", action="store_const",
  help="Inverse", default=False)



  options, args = parser.parse_args()

  args = parser.parse_args()
  print(calculate_sum_time(options.config, options.source, options.dest, options.arrival, reverse=options.inverse))