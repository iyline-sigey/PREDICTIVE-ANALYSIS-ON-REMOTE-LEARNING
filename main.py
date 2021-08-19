import streamlit as st
import pandas as pd
import seaborn as sns

#Modelling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, accuracy_score, classification_report, f1_score,r2_score
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

header = st.beta_container()
desc = st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
plot = st.beta_container()
prediction = st.beta_container()
modelling = st.beta_container()



# st.markdown(
# 	""""
# 	<style> 
# 	.main {background-color: #5F5F5F}
# 	</style>
# 	""",
# 	unsafe_allow_html=True
# 	)



@st.cache
def get_data(filename):
	df=pd.read_csv(filename)

	return df

with header:
    st.title('Remote Learning ')
    st.write("""# Sentiment  Classification""")
    st.text('Classification Algorithms')
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.image('e_learns.jpg')
    #     st.button('Image 1')

    #model = st.sidebar.selectbox('Select Classification Model',("KNN","Random forest","Gradient Boost Classifier","Naive bayes"))
    #st.write('Classification model : ',model)
	

with dataset:
	st.header('Remote Learning Dataset')
	st.write("This dataset was scraped from twitter API using keywords such as online classes and distant learning to gather information on different perceptions of Remote Learning in Kenya. ")


	df=get_data('remote_clean.csv')
	st.write(df.head())
	st.write('Shape of dataset',df.shape)
	st.write('Numberr of classes',len(df.columns))

	st.subheader("Sentiments from the tweets")
	sentiment=pd.DataFrame(df['sentiment'].value_counts())
	st.bar_chart(sentiment)

	

with features:
	st.header('Classification Algorithms')
	model = st.sidebar.selectbox('Select Classifier: ',("KNN","Random forest","Gradient Boost Classifier","Naive bayes"))
	st.write('Classification model : ',model)



with modelling:
	st.header('Train the model:')

	# def add_parameter(clf_name):
	# 	params =dict()
	# if model == "KNN":
	# 	n_estimators = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)
	# 	params['n_estimators'] =n_estimators
	# elif model == "Random forest":
	# 	max_depth=st.sidebar.slider('Which should be the max_depth of the model?', min_value=10,max_value=100,value=20,step=10)
	# 	n_estimators = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)

	# 	params['max_depth'] =max_depth
	# 	params['n_estimators'] =n_estimators
	# else:
	# 	max_depth=st.sidebar.slider('Which should be the max_depth of the model?', min_value=10,max_value=100,value=20,step=10)
	# 	n_estimators = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)
	# 	learning_rate=st.sidebar.slider("which learning_rate should be used?",min_value=0.01,max_value=0.10,value=0.01,step=0.02)
	# 	params['max_depth'] =max_depth
	# 	params['n_estimators'] =n_estimators
	# 	params['learning_rate'] =learning_rate

	# add_parameter(model)
	sel_col,disp_col=st.beta_columns(2)
	#model = st.sidebar.selectbox("Select a classification Model?", options=['KNN',"Random forest","Gradient Boost Classifier","Naive bayes"], index=0)
	
	

	max_depth=st.sidebar.slider('Which should be the max_depth of the model?', min_value=10,max_value=100,value=20,step=10)
	n_estimators = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)
	alphas=st.sidebar.selectbox("which alphas should be used?",options=[0.1, 0.001, 0.2, 0.3, 0.4, 0.5],index=0)
	learning_rate=st.sidebar.slider("which learning_rate should be used?",min_value=0.01,max_value=0.10,value=0.01,step=0.02)

	#Tokenization
	# set a vocabulary size. This is the maximum number of words that can be used.
	vocabulary_size = 10000
	max_words = 5000
	max_len = 200

	X=df.clean_tweet.values
	y=df.sentiment.values

	from sklearn.preprocessing import LabelEncoder
	le=LabelEncoder()
	y=le.fit_transform(y)

	#  Split the data into train and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	
	# create the tokenizer that comes with Keras.
	tokenizer = Tokenizer(num_words=vocabulary_size)
	tokenizer.fit_on_texts(X_train)

	#convert the texts to sequences.
	X_train_seq = tokenizer.texts_to_sequences(X_train)
	X_val_seq = tokenizer.texts_to_sequences(X_test)

	X_train_seq_padded = pad_sequences(X_train_seq, maxlen=200)
	X_val_seq_padded  = pad_sequences(X_val_seq, maxlen=200)

	if model =="KNN":
		cl=KNeighborsClassifier(n_neighbors=n_estimators)
		cl.fit(X_train_seq_padded,y_train)
		#n_neighbors = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)

		#we will predict our model
		pred =cl.predict(X_val_seq_padded)

	elif model=="Random forest":
		#Random forest
		forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
		# max_depth=st.sidebar.slider('Which should be the max_depth of the model?', min_value=10,max_value=100,value=20,step=10)
		# n_estimators = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)

		# Train it on our training set.
		forest.fit(X_train_seq_padded,y_train)

		# Predict based on the model we've trained
		pred = forest.predict(X_val_seq_padded)

	elif model=='Gradient Boost Classifier':
		gbr = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

		# max_depth=st.sidebar.slider('Which should be the max_depth of the model?', min_value=10,max_value=100,value=20,step=10)
		# n_estimators = st.sidebar.selectbox("How many trees should there be?", options=[5,10,15], index=0)
		# learning_rate=st.sidebar.slider("which learning_rate should be used?",min_value=0.01,max_value=0.10,value=0.01,step=0.02)

		gbr = gbr.fit(X_train_seq_padded,y_train)
		pred = gbr.predict(X_val_seq_padded)

	else:
		model = MultinomialNB()
		#alphas=st.sidebar.selectbox("which alphas should be used?",options=[0.1, 0.001, 0.2, 0.3, 0.4, 0.5],index=0)
		model.fit(X_train_seq_padded, y_train)
		pred= model.predict(X_val_seq_padded)
	
	

	sel_col.subheader("Your accuracy_score of the model is:")
	sel_col.write(accuracy_score(y_test,pred))

	sel_col.subheader("Your squared error of the model is:")
	sel_col.write(mean_squared_error(y_test,pred))

	sel_col.subheader("Your R2 score of the model is:")
	sel_col.write(r2_score(y_test,pred))



	# model = Sequential()
	# model.add(layers.Embedding(max_words, 20)) #The embedding layer
	# model.add(layers.LSTM(15,dropout=0.5)) #Our LSTM layer
	# model.add(layers.Dense(1,activation='sigmoid'))
	# print(model.summary())
	# model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

	# history = model.fit(X_train_seq_padded, y_train, epochs=10,batch_size=32,
	#                       validation_data=(X_val_seq_padded, y_test))

	# from keras.callbacks import EarlyStopping
	# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
	# # history=model1.fit(X_train, ytrain,
	# #  batch_size=128,
	# #  epochs=20,
	# #  validation_data=[X_test, ytest],
	# #  callbacks=[es])
	# #We save this model so that we can use in own web app
	# model.save('movie_sent.h5')



	

