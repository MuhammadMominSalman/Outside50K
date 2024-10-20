from diffusers import DiffusionPipeline
import torch
import pandas as pd

seasons = ["winter", "summer", "spring", "autumn"]
times = ["day", "night"]
weathers = ["foggy", "rainy", "snowy", "sunny", "cloudy"]

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

arr = ["img_path", "season", "time", "weather"] 
df = pd.DataFrame(columns=arr)

for season in seasons:
    for time in times:
        for weather in weathers:
            for i in range(100):
                img = pipeline("A photo of {s} season and {w} weather and {d} time".format(s=season, w=weather, d=time)).images[0]
                img.save(".\imgs\{}.jpg".format(i))
                df.append({"img_path": "{}.jpg", "season": "{season}", "time": "{time}", "weather":"{weather}"}, ignore_index=True)


df.to_csv(".\labels\labels.csv")


