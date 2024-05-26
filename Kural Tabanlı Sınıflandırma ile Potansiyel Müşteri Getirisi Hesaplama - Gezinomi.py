#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Gezinomi yaptığı satışların bazı özelliklerini kullanarak seviye tabanlı (level based) yeni satış tanımları
# oluşturmak ve bu yeni satış tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.
# Örneğin: Antalya’dan Herşey Dahil bir otele yoğun bir dönemde gitmek isteyen bir müşterinin ortalama ne kadar kazandırabileceği belirlenmek isteniyor.
#############################################
# PROJE GÖREVLERİ
#############################################
# GENEL BILGILENDIRMELER
#############################################

# Veri setini okutma ve veri seti ile ilgili genel bilgiler
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
file_path = r"C:\Users\ExtraBT\Desktop\MASAUSTU\Yazılım\Python\Python Programming for Data Science\gezinomi.xlsx"
df = pd.read_excel(file_path)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(df.head())
print(df.shape)
print(df.info())

# Kaç unique şehir vardır? Frekansları nedir?
print(df["SaleCityName"].nunique())
print(df["SaleCityName"].value_counts())

# Kaç unique Concept vardır?
df["ConceptName"].nunique()

# Hangi Concept'dan kaçar tane satış gerçekleşmiş?
df["ConceptName"].value_counts()

# Şehirlere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("SaleCityName").agg({"Price": "sum"})

# Concept türlerine göre göre ne kadar kazanılmış?
df.groupby("ConceptName").agg({"Price": "sum"})

# Şehirlere göre PRICE ortalamaları nedir?
df.groupby(by=['SaleCityName']).agg({"Price": "mean"})

# Conceptlere göre PRICE ortalamaları nedir?
df.groupby(by=['ConceptName']).agg({"Price": "mean"})

# Şehir-Concept kırılımında PRICE ortalamaları nedir?
df.groupby(by=["SaleCityName", 'ConceptName']).agg({"Price": "mean"})


#############################################
# satis_checkin_day_diff değişkenini EB_Score adında yeni bir kategorik değişkene çevirelim.
#############################################
bins = [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()]
labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins, labels=labels)
df.head(50).to_excel("eb_scorew.xlsx", index=False)

#############################################
# Şehir,Concept, [EB_Score,Sezon,CInday] kırılımında ücret ortalamalarına ve frekanslarına bakalim.
#############################################
# Şehir-Concept-EB Score kırılımında ücret ortalamaları
df.groupby(by=["SaleCityName", 'ConceptName', "EB_Score" ]).agg({"Price": ["mean", "count"]})

# Şehir-Concept-Sezon kırılımında ücret ortalamaları
df.groupby(by=["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": ["mean", "count"]})

# Şehir-Concept-CInday kırılımında ücret ortalamaları
df.groupby(by=["SaleCityName", "ConceptName", "CInDay"]).agg({"Price": ["mean", "count"]})


#############################################
# City-Concept-Season kırılımın çıktısını PRICE'a göre sıralayalım.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayalim.
# Çıktıyı agg_df olarak kaydediyoruz.

agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)
agg_df.head(20)

#############################################
# Indekste yer alan isimleri değişken ismine çevirelim.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çevirelim.
# İpucu: reset_index()
agg_df.reset_index(inplace=True)

agg_df.head()
#############################################
# Yeni level based satışları tanımlayalım ve veri setine değişken olarak ekleyelim.
#############################################
# sales_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyelim.
agg_df['sales_level_based'] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)


#############################################
# Personaları segmentlere ayıralım.
#############################################
# PRICE'a göre segmentlere ayıralım,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyelim
# segmentleri betimleyelim
agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "sum"]})

#############################################
#  Oluşan son df'i price değişkenine göre sıralayalım.
# "ANTALYA_HERŞEY DAHIL_HIGH" hangi segmenttedir ve ne kadar ücret beklenmektedir?
#############################################
agg_df.sort_values(by="Price")

new_user = "ANTALYA_HERŞEY DAHIL_HIGH"
agg_df[agg_df["sales_level_based"] == new_user]

# BURADA ÖRNEK OLARAK ANTALYA HER ŞEY DAHİL İÇİN
# B SEGMENT
# DOLU
# ORTALAMA FİYAT 64.92
# İSTEĞE GÖRE BU İSTEKLER ÇOĞALTILABİLİR