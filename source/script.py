from diffusers import StableDiffusionPipeline
import torch
import requests
from bs4 import BeautifulSoup
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

#model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# for the first time use model_id 
# then use  pipe.save_pretrained("model")
# for the subsequent iterations "model" runs from local machine

pipe = StableDiffusionPipeline.from_pretrained("model")
#pipe.save_pretrained("model")
pipe.to("cuda")


# code to get names of the countries 
url = 'https://www.worldometers.info/geography/alphabetical-list-of-countries/'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
t=soup.find('table', attrs={'class': 'table table-hover table-condensed'})
t_body = t.find('tbody')
#print(t_body)
rows = t_body.find_all('tr')
countries = []
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    countries.append([ele for ele in cols if ele])
#print(countries)
country_list = []
prompt_list=[]



# generate some nice prompt
for i in countries:
    country_list.append(i[1])
    prompt_list.append(i[1]+"world cup football, national flag on the ball, round, complete ball, matchday ball,perfect shape, highest resolution,best,use flag colors, ball covers 80 percent image size")

for i in range(len(country_list)):
    image = pipe(prompt_list[i]).images[0]

    image.save("./footballv3"+country_list[i]+".jpg")
    txt = country_list[i] # prints country name on the top write corner
    
    txtColor = (255, 255, 255) # text color
    txtCoord = (0, 0) # place the text here 
    # please give the path of the font from local machine, on ubuntu distro you might find fonts at the same path
    
    txtFont=ImageFont.FreeTypeFont(font="/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", size=35)
    # save images
    img=Image.open("./football"+country_list[i]+".jpg")
    ImageDraw.Draw(img).text(txtCoord, txt, font=txtFont, fill=txtColor)
    
    # save in webp format
    img.save("./football"+country_list[i]+".webp",quality=95)
