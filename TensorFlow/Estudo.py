
# coding: utf-8

# In[94]:


import tensorflow as tf
 
#Usaremos o pandas para algumas transformações basicas
import pandas as pd
 
#Numpy apenas para deixar as coisas mais rapidas
import numpy as np
 
#Irei utilizar essa função do sklearn para dividir o conjunto de treino e teste
from sklearn.cross_validation import train_test_split
 
#As metricas básica acuracia e confusio_matrix
from sklearn.metrics import confusion_matrix,accuracy_score


import seaborn as sns

from bokeh.sampledata.iris import flowers as dados



data=dados

#Definimos essa função para normalizar o coluna Species
#Exemplo [preto,branco,preto,azul] => [0,1,0,2]
def normalizeIris(x):
    if x == "virginica":
        return 0
    elif x == "versicolor":
        return 1
    else:
        return 2
 
#fazemos a função para transforma as labels em one-hot vectors
#Exemplo [1,2,1,0] => [[0,1,0],[0,0,1],[0,1,0],[1,0,0]]
def makeHotvector(y_data):
    labels = []
    for x in range(len(y_data)):
        labels.append([0,0,0])
        labels[x][y_data[x]] = 1
    y_data = np.array(labels,dtype=np.float64)
    return y_data



#Usamos o método .apply do Pandas para aplicar a função normaLizeIris
#Na coluna "Species" lembre o que essa função faz
data["species"] = data["species"].apply(normalizeIris)
 
#Aqui separamos as features em uma variavel e as labels em outra
x_data = data.drop("species",axis=1).values
y_data = data["species"].values
 
#Aqui separamos um pouco do dataset para treino e a outra parte para teste
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)


# In[95]:


#A quantidade inputs na rede
n_input = 4
#A proporção em que a rede irá atualizar cada peso
#Chamamos também de alpha
learning_rate = 0.01
# A quantidade de unidades no primeiro layer
n_hidden_unites_1 = 5
#A quantidade de unidades no segundo layer
n_hidden_unites_2 = 3

#Vamos criar o grapho da rede
#Primeiro definimos a sessão
sess = tf.Session()

#Aqui são os inputs da rede
#Apenas os inputs do dataset são considerados.
X = tf.placeholder(shape=[None,n_input],dtype=tf.float64)
 
#Aqui ficam as Labels para treino
#Essa parte do grafo não aparece no esquema.
#mas aqui fica as labels que serão usadas como exemplo para a rede.
y_ =  tf.placeholder(dtype=tf.float64)


#Agora iremos definir os pesos da rede
#Esse é nosso primeira camada de pesos.
W = {"h1":tf.Variable(tf.random_normal([n_input,n_hidden_unites_1],dtype=tf.float64),dtype=tf.float64),
     "h2":tf.Variable(tf.random_normal([n_hidden_unites_1,n_hidden_unites_2],dtype=tf.float64),dtype=tf.float64)
    }

#Esse é o bias da rede
b = {"b1":tf.Variable(tf.random_normal([n_hidden_unites_1],dtype=tf.float64),dtype=tf.float64),
     "b2":tf.Variable(tf.random_normal([n_hidden_unites_2],dtype=tf.float64),dtype=tf.float64)
    }

#Sigmoide aplicado a rede
#O output do hidden_1
out_hidden_1 = tf.nn.sigmoid(tf.matmul(X,W["h1"])+b["b1"])
#o output do hidden_2 que alias será o usado como nosso output.
out_hidden_2 = tf.nn.sigmoid(tf.matmul(out_hidden_1,W["h2"])+b["b2"])

#Definimos o erro quadratico do out_hidden_2 em relação y_
#isso permite deriva-lo e depois ajustar os pesos
mse = tf.losses.mean_squared_error(y_,out_hidden_2)
 
#Aqui eu uso o método gradientDescent para otimizar o parametro mse
#Ele ira derivar o erro em relação a cada pesso da rede e fazer um atualização
#para cima ou para abaixo proporcional ao learning_hate da rede
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)


#Aqui iremos inicializar todas as variaveis que compõe o grafo.

#Uma maneira de carregar as variaveis para a memoria.
init = tf.global_variables_initializer()
sess.run(init)


# In[ ]:


#Realizar 1000 interações no dataset

from tqdm import tqdm #Barra de progresso

for x in tqdm(range(1000)):
    #print(x)#para ver o andamento do treino
    
    #Iremos dar loop do tamanho do dataset
    for _ in range(len(x_train)):
        #Aqui seleciono uma instancia aliatória do dataset
        i = np.random.randint(len(x_train))
        x = np.array([x_train[i]])
        y = np.array([y_train[i]])
 
        #Aqui treinamos a rede uma iteração de cada vez
        train_step.run(feed_dict={X:x,y_:y},session=sess)


# In[ ]:


#Aqui iremos medir a precisão do modelo
#colocamos a rede para validar os outputs no conjunto de teste
outputs = out_hidden_2.eval(feed_dict={X:x_test},session=sess)
 
#Aqui convertemos de one-hot-vector para discreto
labels = [x.argmax() for x in y_test]
predictions = [x.argmax() for x in outputs]
 
#aplicamos confusion matrix para ver o desepenho
cm = confusion_matrix(labels,predictions)
 
#aqui vemos a acuracia do algoritmo.
acuracy = accuracy_score(labels,predictions)
 
#print(cm)
print("Aproximadamento %0.2f%%" % (acuracy*100))

#print(labels)

