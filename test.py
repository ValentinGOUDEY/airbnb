from PIL import Image
import numpy as np
import pandas as pd

import streamlit as st

import altair as alt
import seaborn as sns

import datetime


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
st.set_option('deprecation.showPyplotGlobalUse', False)




with dataset:
	airbnb = st.cache(pd.read_csv)("airbnb_cleaned.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	df_paris= st.cache(pd.read_csv)("df_paris.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	df_london= st.cache(pd.read_csv)("df_london.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	df_superhost = st.cache(pd.read_csv)("df_superhost.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	tab = pd.DataFrame([[3120,351,3,1],[100,2478,728,99],[0,854,1820,751],[0,151,795,2524]])
	tab2 = pd.DataFrame([[0.832,0.023],[0.071,0.073]])
	df=pd.read_csv('df.csv')

	y_pred = pd.read_csv('y_pred_knn.csv')
	y_pred_superhost = pd.read_csv('y_pred_rf.csv')
	##st.write(df_superhost.head())
	
	################################ configuration de page ################################
	#apptitle = "Projet AirBnb"
	#st.set_page_config(page_title=apptitle, page_icon="🏠")



	################################ SIDEBAR ################################
	image = Image.open('logo-Airbnb.png')
	st.sidebar.image(image)

	st.sidebar.header("Bienvenue!")

	st.sidebar.markdown(" ")
	st.sidebar.markdown("*Nous sommes 4 étudiants en Datasciences chez Datascientest et nous travaillons sur ce projet Airbnb afin de valider notre diplome de Data Analyst.*")

	st.sidebar.markdown('-----------------------------------------------------')
	
	st.sidebar.header('Sélectionnez une ville pour afficher la carte intéractive')

	check = st.sidebar.checkbox("Paris")
	st.sidebar.write('Carte Superhost Paris active:', check)

	
	check_2 = st.sidebar.checkbox("Londres")
	st.sidebar.write("Carte Superhost Londres active:", check_2)
	

	#values = st.sidebar.slider("Tranche de prix (€)", float(df_paris.price.min()), float(df_paris.price.clip(upper=10000000.).max()), (8., 999.))
	#min_nights_values = st.sidebar.slider('minimum_nights', 0, 30, (1))
	reviews = st.sidebar.slider('Number_of_reviews', 8, 364, (8))
	
	#-- Set property type
	#select_property = st.sidebar.multiselect('Quel type de logement ?',df_superhost.property_type.unique())     #applique aucun changement :/

	#select_appart = st.sidebar.selectbox('id')

	st.sidebar.markdown('-----------------------------------------------------')
	
	st.sidebar.header("Filtre model prédiction")

	#st.sidebar.markdown("**Author**: Valentin Goudey")


	#st.sidebar.markdown("**Version:** 1.0.0")



	################################ SOMMAIRE ################################

	st.title("Projet Airbnb Datascientest")

	st.markdown('-----------------------------------------------------')

	st.markdown("**Dataset d’origine:** *Dataset airbnb rassemblant des données de logements dans de grandes villes tout autour du monde via le site opendatasoft nettoyé pour garder uniquement les villes de Paris et Londres*")


	#################### centre d'interet ######################

	st.header("Qu’est-ce que tu cherches ?")
	st.write(f"Dans les colonnes, vous pourriez vouloir afficher uniquement un sous-ensemble.")
	st.markdown("_**Note:** Il est possible de filtrer nos données de manière plus conventionnelle en utilisant les caractéristiques suivantes : **Prix**, **Minimum de nuits**, **Type de chambre**, **Quartiers**, **Description de l'hote**, **Nombre de Commentaires**")
	defaultcols = ["price", "minimum_nights", "room_type", "neighbourhood_cleansed", "description", "number_of_reviews"]
	cols = st.multiselect('', airbnb.columns.tolist(), default=defaultcols)
	st.dataframe(airbnb[cols].head(40))

	################### Pourcentage de distribution par Ville #####################

	st.header("Disponibilitée et distribution par Ville")
	st.markdown("La variable **availability_365** indique le nombre de jours dans l'année (365) où le bien est disponible.")

	city_select = st.radio("Ville", df_superhost.city.unique())
	
	@st.cache
	def get_availability(show_exp):
		return df_superhost.query(f"""city==@city_select\
			and availability_365>0""").availability_365.describe(\
				percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T

	st.table(get_availability(city_select))


	################### Cartes intéractive Superhost #####################

	if check:
		st.map(df_paris.query(f"review_scores_rating>=96 and number_of_reviews>=8 and host_response_rate>=90 and number_of_reviews>={reviews}")[["latitude", "longitude"]].dropna(how="any"))

	if check_2:
		st.map(df_london.query(f"review_scores_rating>=96 and number_of_reviews>=8 and host_response_rate>=90 and number_of_reviews>={reviews}")[["latitude", "longitude"]].dropna(how="any"))

	################### model de prédiction #####################
	
	st.markdown('-----------------------------------------------------')
	
	st.header("Modèles de prédiction")
	
	st.subheader("Les résultats obtenus pour notre modèle de classification pour le succès d'un bien :")
	st.write(tab)
	st.write("Ce qui nous donne une accuracy de 72%")

	st.subheader("Les résultats obtenus pour notre modèle de classification pour le superhost :")
	st.write(tab2)
	st.write("Ce qui nous donne une accuracy de 90%")


	select_appart = st.sidebar.selectbox('Quel bien ?',df["Id"])

	st.markdown("### Exemple de notre modèle sur le bien sélectionné")
	st.write("Vu d'ensemble du bien sélectionné")
	data_ex = df[df["Id"]==select_appart]
	st.write(data_ex)
	
	st.markdown('-----------------------------------------------------')

	st.write(f"##### Le succès de ce bien est classé: {int(df['succes'].loc[df['Id']==select_appart])} ------> Le succès du bien trouvé par l'algorithme est: {int(y_pred['0'][df.index[df['Id']==select_appart]])}")

	def ouinon(k):
			if int(k)==0:
				return("Non")
			if int(k)==1:
				return("Oui")
			
	st.markdown('-----------------------------------------------------')
		
	st.write(f"##### L'hôte est-il un superhost?: {ouinon(int(df['superhost'].loc[df['Id']==select_appart]))} ------> Notre algorithme prédit-il notre hote comme Superhost : {ouinon(int(y_pred_superhost['0'][df.index[df['Id']==select_appart]]))}")

	st.markdown('-----------------------------------------------------')
