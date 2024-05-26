
##################################################
# List Comprehensions
##################################################

# ###############################################
# # List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
# ###############################################
#
# # Beklenen Çıktı
#
# # ['NUM_TOTAL',
# #  'NUM_SPEEDING',
# #  'NUM_ALCOHOL',
# #  'NUM_NOT_DISTRACTED',
# #  'NUM_NO_PREVIOUS',
# #  'NUM_INS_PREMIUM',
# #  'NUM_INS_LOSSES',
# #  'ABBREV']
#
# # Notlar:
# # Numerik olmayanların da isimleri büyümeli.
# # Tek bir list comp yapısı ile yapılmalı.


import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()


["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]


# ###############################################
# # List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazalım.
# ###############################################
#
# # Notlar:
# # Tüm değişken isimleri büyük olmalı.
# # Tek bir list comp ile yapılmalı.
#
# # Beklenen çıktı:
#
# # ['TOTAL_FLAG',
# #  'SPEEDING_FLAG',
# #  'ALCOHOL_FLAG',
# #  'NOT_DISTRACTED',
# #  'NO_PREVIOUS',
# #  'INS_PREMIUM_FLAG',
# #  'INS_LOSSES_FLAG',
# #  'ABBREV_FLAG']


[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

# ###############################################
# # List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçelim ve yeni bir dataframe oluşturalım.
# ###############################################
#
og_list = ["abbrev", "no_previous"]
#
# # Notlar:
# # Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturalım.
# # Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturalım adını new_df olarak isimlendirelim.
#
# # Beklenen çıktı:
#
# #    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# # 0 18.800     7.332    5.640          18.048      784.550     145.080
# # 1 18.100     7.421    4.525          16.290     1053.480     133.930
# # 2 18.600     6.510    5.208          15.624      899.470     110.350
# # 3 22.400     4.032    5.824          21.056      827.340     142.390
# # 4 12.000     4.200    3.360          10.920      878.410     165.630
#

og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df.head()





##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayalım.
#########################################
df = sns.load_dataset("titanic")
df.head()
df.shape

#########################################
# Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulalım.
#########################################

df["sex"].value_counts()


#########################################
# Her bir sutuna ait unique değerlerin sayısını bulalım.
#########################################

df.nunique()

#########################################
# pclass değişkeninin unique değerleri bulalım.
#########################################

df["pclass"].unique()


#########################################
# pclass ve parch değişkenlerinin unique değerlerinin sayısını bulalım.
#########################################

df[["pclass","parch"]].nunique()

#########################################
# embarked değişkeninin tipini kontrol edelim. Tipini category olarak değiştirelim. Tekrar tipini kontrol edelim.
#########################################

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()


#########################################
# embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"]=="C"].head(10)


#########################################
# embarked değeri S olmayanların tüm bilgelerini gösterelim.
#########################################
df[df["embarked"] != "S"].head(10)

df[df["embarked"] != "S"]["embarked"].unique()

df[~(df["embarked"] == "S")]["embarked"].unique()


#########################################
# Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"]<30) & (df["sex"]=="female")].head()


#########################################
# Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["fare"] > 500 ) | (df["age"] > 70 )].head()


#########################################
# Her bir değişkendeki boş değerlerin toplamını bulualım.
#########################################

df.isnull().sum()


#########################################
# who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1, inplace=True)


#########################################
# deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurualım.
#########################################


type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()


#########################################
# age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()

#########################################
# survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulalım.
#########################################

df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})


#########################################
# 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazalım.
# Yazdığalım fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturalım oluşturalım. (apply ve lambda yapılarını kullanalım)
#########################################

def age_30(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))


df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)


#########################################
# Seaborn kütüphanesi içerisinden Tips veri setini tanımlayalım.
#########################################

df = sns.load_dataset("tips")
df.head()
df.shape


#########################################
# Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulalım.
#########################################

df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

#########################################
# Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulalım.
#########################################

df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})

#########################################
# Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulalım.
#########################################


df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip":  ["sum","min","max","mean"],
                                                                            "Lunch" : lambda x:  x.nunqiue()})
#########################################
# size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################


df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean() # 17.184965034965035


#########################################
# total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()


#########################################
# total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayalım ve ilk 30 kişiyi yeni bir dataframe'e atayalım.
#########################################

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape
