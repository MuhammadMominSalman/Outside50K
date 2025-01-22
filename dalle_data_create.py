import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import openai


openai.api_key = "  "


seasons = ["winter", "summer", "spring", "autumn"]
times = ["day", "night"]
weathers = ["foggy", "rainy", "snowy", "sunny", "cloudy"]

response = openai.Image.create(
    prompt = "A caricature of a cute happy puppy.",
    n = 1,
    size = "512x512"
)

arr = ["img_path", "season", "time", "weather", "mode"] 
df = pd.DataFrame(columns=arr)

for season in seasons:
    for time in times:
        for weather in weathers:
            mode = "train"
            for i in range(10):
                prompt =  "A photo of {s} season and {w} weather and {d} time".format(s=season, w=weather, d=time)
                url = openai.Image.create(
                prompt = prompt,
                n = 1,
                size = "512x512")["data"][0]["url"]
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img.save(f"./imgs/{season}_{weather}_{time}_{i}.jpg")
                if i >= 8:
                    mode = "test"
                df = df._append({"img_path": f"{season}_{weather}_{time}_{i}.jpg", 
                                 "season": f"{season}", 
                                 "time": f"{time}", 
                                 "weather": f"{weather}",
                                 "mode":mode}, ignore_index=True)


df.to_csv("./labels/labels.csv")
