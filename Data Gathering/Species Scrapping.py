import requests
from bs4 import BeautifulSoup

NA_Species_URL = 'https://ebird.org/region/na?yr=all&m=&rank=mrec'
NA_Species_Website = requests.get(NA_Species_URL)
NA_Species_Soup = BeautifulSoup(NA_Species_Website.text, 'lxml')
Species_Table = NA_Species_Soup.find('table', id='sightingsTable')
Species_Rows = Species_Table.findAll('tr', {'class': 'has-details'})
Species = []
Refs = []
i = 0
for x in Species_Rows:
    Species_Code = x.find('td', {'class': 'species-name'})
    if Species_Code is None:
        continue
    if len(Species_Code.contents) == 1:
        continue

    Ref_and_Name = Species_Code.find('a')
    Ref = str(Ref_and_Name.attrs['href'])
    Name = str(Ref_and_Name.contents[0])
    Species.append(Name)
    Refs.append(Refs)




