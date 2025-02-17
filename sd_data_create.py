from diffusers import DiffusionPipeline
import torch
import pandas as pd

IMG_PER_CAT = 1000

seasons = ["winter", "summer", "spring", "autumn"]
times = ["day", "night"]
weathers = ["foggy", "rainy", "snowy", "sunny", "cloudy"]

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

arr = ["img_path", "season", "time", "weather", "mode"] 
df = pd.DataFrame(columns=arr)

for season in seasons:
    for time in times:
        for weather in weathers:
            mode = "train"
            for i in range(IMG_PER_CAT):
                prompt =  "A photo of {s} season and {w} weather and {d} time".format(s=season, w=weather, d=time)
                img = pipeline(prompt).images[0]
                img.save(f"/mnt/Data/musa7216/seasonal_synthetic/{season}_{weather}_{time}_{i}.jpg")
                if i >= IMG_PER_CAT*0.9:
                    mode = "test"
                df = df._append({"img_path": f"{season}_{weather}_{time}_{i}.jpg", 
                                 "season": f"{season}", 
                                 "time": f"{time}", 
                                 "weather": f"{weather}",
                                 "mode":mode}, ignore_index=True)


df.to_csv("/mnt/Data/musa7216/seasonal_synthetic/labels.csv")
df.to_csv("./labels/labels.csv")


