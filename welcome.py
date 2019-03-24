import pandas as pd
import numpy as np
energy=pd.DataFrame()
GDP=pd.DataFrame()
ScimEn=pd.DataFrame()

def answer_one():
    #skipping header
    energy = pd.read_excel('Energy Indicators.xls',skiprows=17)
    #drop first two columns
    energy.drop(energy.columns[0:2],axis=1,inplace=True)
    #skip footer
    energy=energy[:227]
    #assigning columns
    energy.columns=['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

    #replacing '...' values with np.NaN
    energy = energy.replace("...",np.NaN)


    #Convert Energy Supply to gigajoules
    energy['Energy Supply']=1000000*energy['Energy Supply']

    #replacing specified country names
    '''"Republic of Korea": "South Korea",
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "China, Hong Kong Special Administrative Region": "Hong Kong"'''

    energy['Country']=energy['Country'].str.replace("Republic of Korea","South Korea").str.replace('United States of America','United States').str.replace("United Kingdom of Great Britain and Northern Ireland","United Kingdom").str.replace("China, Hong Kong Special Administrative Region", "Hong Kong")


    #removing numbers and parathesis info from counries


    for i in range(len(energy['Country'])):
        index=energy['Country'].iloc[i].find('(')
        energy['Country'].iloc[i]=energy['Country'].iloc[i][0:index-1]


    for i in range(len(energy['Country'])):
        str=''
        for char in energy['Country'].iloc[i]:
            if(char.isdigit()==False):
                str+=char
        energy['Country'].str.replace(energy['Country'].iloc[i],str)
    #reading GDP data

    GDP=pd.read_csv('world_bank.csv',encoding='unicode_escape',skiprows=4)
    #replacing country names
    '''"Korea, Rep.": "South Korea", 
    "Iran, Islamic Rep.": "Iran",
    "Hong Kong SAR, China": "Hong Kong"
    
    '''
    GDP['Country Name']=GDP['Country Name'].str.replace("Korea, Rep.", "South Korea").str.replace("Iran, Islamic Rep.", "Iran").str.replace("Hong Kong SAR, China", "Hong Kong")
    ScimEn=pd.read_excel('scimagojr-3.xlsx')
    #merging
    x=pd.merge(ScimEn,energy,left_on='Country',right_on='Country',how ='outer')
    df = pd.merge(x,GDP,left_on='Country',right_on='Country Name',how ='outer')
    df = df[['Country','Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    print(df.columns)
    df.set_index(['Country'],inplace =True)
    df = df[df['Rank'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])]
    #sorting accordingly
    df.sort_values(by='Rank', ascending=True, inplace=True)
    return df


def answer_two():
    energy = pd.read_excel('Energy Indicators.xls', skiprows=17)
    # drop first two columns
    energy.drop(energy.columns[0:2], axis=1, inplace=True)
    # skip footer
    energy = energy[:227]
    # assigning columns
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

    # replacing '...' values with np.NaN
    energy = energy.replace("...", np.NaN)

    # Convert Energy Supply to gigajoules
    energy['Energy Supply'] = 1000000 * energy['Energy Supply']

    # replacing specified country names
    '''"Republic of Korea": "South Korea",
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "China, Hong Kong Special Administrative Region": "Hong Kong"'''

    energy['Country'] = energy['Country'].str.replace("Republic of Korea", "South Korea").str.replace(
        'United States of America', 'United States').str.replace("United Kingdom of Great Britain and Northern Ireland",
                                                                 "United Kingdom").str.replace(
        "China, Hong Kong Special Administrative Region", "Hong Kong")

    # removing numbers and parathesis info from counries

    for i in range(len(energy['Country'])):
        index = energy['Country'].iloc[i].find('(')
        energy['Country'].iloc[i] = energy['Country'].iloc[i][0:index - 1]

    for i in range(len(energy['Country'])):
        str = ''
        for char in energy['Country'].iloc[i]:
            if (char.isdigit() == False):
                str += char
        energy['Country'].str.replace(energy['Country'].iloc[i], str)
    # reading GDP data

    GDP = pd.read_csv('world_bank.csv', encoding='unicode_escape', skiprows=4)

    # replacing country names
    '''"Korea, Rep.": "South Korea", 
    "Iran, Islamic Rep.": "Iran",
    "Hong Kong SAR, China": "Hong Kong"

    '''
    GDP['Country Name'] = GDP['Country Name'].str.replace("Korea, Rep.", "South Korea").str.replace(
        "Iran, Islamic Rep.", "Iran").str.replace("Hong Kong SAR, China", "Hong Kong")
    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    # outer join
    x = pd.merge(ScimEn, energy, left_on='Country', right_on='Country', how='outer')
    outer = pd.merge(x, GDP, left_on='Country', right_on='Country Name', how='outer')
    # inner join
    y = pd.merge(ScimEn, energy, left_on='Country', right_on='Country', how='inner')
    inner = pd.merge(y, GDP, left_on='Country', right_on='Country Name', how='inner')
    return len(outer) - len(inner)

def answer_three():
    Top15 = answer_one()
    avg = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1,skipna=True).sort_values(ascending=False)
    return avg

def answer_four():
    df = answer_one()
    avg = answer_three()
    df['avg']=avg
    df.sort_index(by = 'avg',ascending = False,inplace = True)
    GDP_change = df.iloc[5]['2015']-df.iloc[5]['2006']
    return(GDP_change)
def answer_five():
    resultdf = answer_one()
    meanenergypercapita = resultdf['Energy Supply per Capita'].mean(axis=0, skipna = False)
    print(type(meanenergypercapita))
    converted_value = getattr(meanenergypercapita, "tolist", lambda x=meanenergypercapita: x)()
    print(type(converted_value))
    return converted_value
print(answer_five())
def answer_six():
    Top15 = answer_one()
    t=Top15.sort_values(by='% Renewable',ascending = False).ix[0]
    return (t.name,t['% Renewable'])
def answer_seven():
    Top15 = answer_one()
    sum = Top15['Self-citations'].sum(axis=0,skipna=True)
    Top15['Self-Citations_ratio']=Top15['Self-citations']/sum
    tuple=Top15.sort_values(by='Self-Citations_ratio',ascending=False).ix[0]
    return (tuple.name,tuple['Self-Citations_ratio'])
def answer_eight():
    Top15 = answer_one()
    Top15['Population_Estimate']= Top15['Energy Supply']/Top15['Energy Supply per Capita']
    tuple=Top15.sort_values(by='Population_Estimate',ascending=False).ix[2]
    return tuple.name
def answer_nine():
    Top15 = answer_one()
    Top15['Population_Estimate'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['Population_Estimate']
    return Top15['Citable docs per Capita'].corr(Top15['Energy Supply per Capita'])

def answer_ten():
    Top15 = answer_one()
    median=Top15['% Renewable'].median(axis=0)
    list=[1 if item>=median else 0 for item in Top15['% Renewable'] ]
    Top15['HighRenew']=list
    Top15.sort_values(by='Rank',ascending=True,inplace=True)
    return Top15['HighRenew']
def answer_eleven():
    Top15 = answer_one()
    Top15['Population_Estimate'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']

    ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}
    #changing the index name
    Top15['Country']=Top15.index
    Top15.rename(index=ContinentDict, inplace=True)
    Top15.index.rename('Continent',inplace=True)
    Top15.reset_index(inplace=True)
    result = Top15[['Continent', 'Population_Estimate']].groupby('Continent').agg(['size', 'sum', 'mean', 'std'])
    return result
def answer_twelve():
    Top15=answer_one()
    Top15['bin'] = pd.cut(Top15['% Renewable'],5)
    ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}
    #changing the name of index
    Top15['Country'] = Top15.index
    Top15.rename(index=ContinentDict, inplace=True)
    Top15.index.rename('Continent', inplace=True)
    Top15.reset_index(inplace=True)
    group = Top15.groupby(['Continent','bin'])
    return group.size()

def answer_thirteen():
    Top15 = answer_one()
    Top15['Population_Estimate'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Population_Estimate']=Top15['Population_Estimate'].apply(lambda x: '{0:,}'.format(x))
    return Top15['Population_Estimate']
print(answer_one())
print(answer_two())
print(answer_three())
print(answer_four())
print(answer_five())
print(answer_six())
print(answer_seven())
print(answer_eight())
print(answer_nine())
print(answer_ten())
print(answer_eleven())
print(answer_twelve())
print(answer_thirteen())