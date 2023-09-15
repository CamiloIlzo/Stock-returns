
# Importação de bibliotecas de programas
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

sequenc = 5 #20 

#----------------------------------------------------
# Vetor de Saída
#----------------------------------------------------
a=0
        
# Segementação do Conjunto de Dados em Treinamento, Validação e Testes 
valid_pct = 10 
test_pct = 10 

indsaida = 0

passosequenc = sequenc-1 
numentradas = 4 
numneurons = 200 
numsaidas = 4
numcamadas = 5
txaprendizado = 0.001
tamlote = 50
#numepocas = 2 # 100 


class RedeLstm(object):
    #==========================================================================
    # Classe RedeLstm lê, padroniza e segmenta dados, configura parâmetros e treina rede LSTM
    #==========================================================================

    def __init__(self):
        print('init RedeLstm')

    #==========================================================================
    # Escala MinMax 
    #==========================================================================
    def escalaminmax(self, df):
        escalaminmax = sklearn.preprocessing.MinMaxScaler()
        df['Open'] = escalaminmax.fit_transform(df.Open.values.reshape(-1,1))
        df['Low'] = escalaminmax.fit_transform(df.Low.values.reshape(-1,1))
        df['High'] = escalaminmax.fit_transform(df.High.values.reshape(-1,1))
        df['Close'] = escalaminmax.fit_transform(df['Close'].values.reshape(-1,1))
        return df
    
    #==========================================================================
    # Escala Retornos 
    #==========================================================================
    def escalaretornos(self, df):
        df['Open'] = df['Open'].pct_change()
        df['Low'] = df['Low'].pct_change()
        df['High'] = df['High'].pct_change()
        df['Close'] = df['Close'].pct_change()
        return df
    
    #==========================================================================
    # Segmenta Dados
    #==========================================================================
    def segmenta(self, acoes, sequenc):
        #matrizcotacoes = acoes.as_matrix() 
        matrizcotacoes = acoes.values
        dadoscotacoes = [] 
        for cont in range(len(matrizcotacoes) - sequenc): 
            dadoscotacoes.append(matrizcotacoes[cont: cont + sequenc])
        dadoscotacoes = np.array(dadoscotacoes)
        valid_tam = int(np.round(valid_pct/100*dadoscotacoes.shape[0]))
        test_tam = int(np.round(test_pct/100*dadoscotacoes.shape[0]))
        trein_tam = dadoscotacoes.shape[0] - (valid_tam + test_tam)
               
        x_trein = dadoscotacoes[:trein_tam,:-1,:]
        y_trein = dadoscotacoes[:trein_tam,-1,:]
        x_valid = dadoscotacoes[trein_tam:trein_tam+valid_tam,:-1,:]
        y_valid = dadoscotacoes[trein_tam:trein_tam+valid_tam,-1,:]
        x_test = dadoscotacoes[trein_tam+valid_tam:,:-1,:]
        y_test = dadoscotacoes[trein_tam+valid_tam:,-1,:]
        
        return [x_trein, y_trein, x_valid, y_valid, x_test, y_test]

    #==========================================================================
    # Segmenta e Normaliza Dados
    #==========================================================================
    def segmentanorm(self, acoes, sequenc):
        #matrizcotacoes = acoes.as_matrix() 
        matrizcotacoes = acoes.values
        dadoscotacoes = [] 
        for cont in range(len(matrizcotacoes) - sequenc): 
            dadoscotacoes.append(matrizcotacoes[cont: cont + sequenc])
        dadoscotacoes = np.array(dadoscotacoes)
        valid_tam = int(np.round(valid_pct/100*dadoscotacoes.shape[0]))
        test_tam = int(np.round(test_pct/100*dadoscotacoes.shape[0]))
        trein_tam = dadoscotacoes.shape[0] - (valid_tam + test_tam)
               
        x_trein = dadoscotacoes[:trein_tam,:-1,:]
        media = dadoscotacoes[:trein_tam,:-1,:].mean(axis=0)
        dvpad = dadoscotacoes[:trein_tam,:-1,:].std(axis=0)
        x_trein -= media
        x_trein /= dvpad

        y_trein = dadoscotacoes[:trein_tam,-1,:]
        media =   dadoscotacoes[:trein_tam,-1,:].mean(axis=0)
        dvpad =   dadoscotacoes[:trein_tam,-1,:].std(axis=0)
        y_trein -= media
        y_trein /= dvpad
        
        x_valid = dadoscotacoes[trein_tam:trein_tam+valid_tam,:-1,:]
        media =   dadoscotacoes[trein_tam:trein_tam+valid_tam,:-1,:].mean(axis=0)
        dvpad =   dadoscotacoes[trein_tam:trein_tam+valid_tam,:-1,:].std(axis=0)
        x_valid -= media
        x_valid /= dvpad

        y_valid = dadoscotacoes[trein_tam:trein_tam+valid_tam,-1,:]
        media =   dadoscotacoes[trein_tam:trein_tam+valid_tam,-1,:].mean(axis=0)
        dvpad =   dadoscotacoes[trein_tam:trein_tam+valid_tam,-1,:].std(axis=0)
        y_valid -= media
        y_valid /= dvpad
        
        x_test = dadoscotacoes[trein_tam+valid_tam:,:-1,:]
        media =  dadoscotacoes[trein_tam+valid_tam:,:-1,:].mean(axis=0)
        dvpad =  dadoscotacoes[trein_tam+valid_tam:,:-1,:].std(axis=0)
        x_test -= media
        x_test /= dvpad
        
        y_test = dadoscotacoes[trein_tam+valid_tam:,-1,:]
        media =  dadoscotacoes[trein_tam+valid_tam:,-1,:].mean(axis=0)
        dvpad =  dadoscotacoes[trein_tam+valid_tam:,-1,:].std(axis=0)
        y_test -= media
        y_test /= dvpad
        
        return [x_trein, y_trein, x_valid, y_valid, x_test, y_test]
    
    #==========================================================================
    # Obtém Lote Seguinte
    #==========================================================================
    def proxlote(self, tamlote):
        #global indsaida, x_trein, matrizrandomiz  
        global indsaida
        inicia = indsaida
        indsaida += tamlote 
        if indsaida > x_trein.shape[0]:
            np.random.shuffle(matrizrandomiz) 
            inicia = 0 
            indsaida = tamlote   
        term = indsaida
        return x_trein[matrizrandomiz[inicia: term]], y_trein[matrizrandomiz[inicia: term]]

    #==========================================================================
    # Treina Rede
    #==========================================================================
    def treinarede(self, numepocas = 2, txaprend=0.001, acao='PETR4'):

        #global indsaida, x_trein, y_trein, matrizrandomiz  
        global x_trein, y_trein, matrizrandomiz  

        #----------------------------------------------------
        # Le arquivo de dados
        #----------------------------------------------------
        nomearq = acao + ".csv"
        #cotacoes = pd.read_csv("G:/Meu Drive/Data/VALE3.csv", index_col = 0) 
        cotacoes = pd.read_csv(nomearq, index_col = 0) 
        cotacoesOHLC = cotacoes.copy()
        cotacoesOHLC = cotacoesOHLC.dropna()
        cotacoesOHLC = cotacoesOHLC[['Close', 'Open', 'Low', 'High']]
        
        print('cotacoesOHLC-Precos')
        print(cotacoesOHLC.head())

        #----------------------------------------------------
        # Padroniza dados
        #----------------------------------------------------
        cotacoesOHLC_norm = cotacoesOHLC.copy()
        
        cotacoesOHLC_norm = objeto.escalaretornos(cotacoesOHLC_norm)
        cotacoesOHLC_norm.dropna(inplace=True)
        #cotacoesOHLC_norm = objeto.escalaminmax(cotacoesOHLC_norm)
        
        print('cotacoesOHLC_norm-Retornos')
        print(cotacoesOHLC_norm.head())

        #----------------------------------------------------
        # Segmenta dos dados
        #----------------------------------------------------
        #x_trein, y_trein, x_valid, y_valid, x_test, y_test = objeto.segmenta(cotacoesOHLC_norm, sequenc)
        x_trein, y_trein, x_valid, y_valid, x_test, y_test = objeto.segmentanorm(cotacoesOHLC_norm, sequenc)

        #----------------------------------------------------
        # Mostra x_trein
        #----------------------------------------------------
        print("")
        print("Mostra x_trein")
        print(x_trein.shape[0])

        print("x_trein[0:10,0]")
        print(x_trein[0:10,0])
                
        #----------------------------------------------------
        # Mostra y_trein
        #----------------------------------------------------
        print("")
        print("Mostra y_trein")
        print(y_trein.shape[0])

        print("y_trein[0:10,0]")
        print(y_trein[0:10,0])
                
        print("y_trein[0:10,1]")
        print(y_trein[0:10,1])
                
        print("y_trein[0:10,2]")
        print(y_trein[0:10,2])
                
        print("y_trein[0:10,3]")
        print(y_trein[0:10,3])
                
        #----------------------------------------------------
        # Mostra x_valid
        #----------------------------------------------------
        print("")
        print("Mostra x_valid")
        print(x_valid.shape[0])

        print("x_valid[0:10,0]")
        print(x_valid[0:10,0])
                
        #----------------------------------------------------
        # Mostra y_valid
        #----------------------------------------------------
        print("")
        print("Mostra y_valid")
        print(y_valid.shape[0])

        print("y_valid[0:10,0]")
        print(y_valid[0:10,0])
                
        print("y_valid[0:10,1]")
        print(y_valid[0:10,1])
                
        print("y_valid[0:10,2]")
        print(y_valid[0:10,2])
                
        print("y_valid[0:10,3]")
        print(y_valid[0:10,3])
                
        #----------------------------------------------------
        # Mostra x_test
        #----------------------------------------------------
        print("")
        print("Mostra x_test")
        print(x_test.shape[0])

        print("x_test[0:10,0]")
        print(x_test[0:10,0])
                
        #----------------------------------------------------
        # Mostra y_test
        #----------------------------------------------------
        print("")
        print("Mostra y_test")
        print(y_test.shape[0])

        print("y_test[0:10,0]")
        print(y_test[0:10,0])
                
        print("y_test[0:10,1]")
        print(y_test[0:10,1])
                
        print("y_test[0:10,2]")
        print(y_test[0:10,2])
                
        print("y_test[0:10,3]")
        print(y_test[0:10,3])
                
        #----------------------------------------------------
        # Mostra Grafico y_test
        #----------------------------------------------------
        comp = pd.DataFrame({'Col3':y_test[:,3]})
        plt.figure(figsize=(10,5))
        plt.plot(comp['Col3'], color='blue', label='y_test')
        plt.title(acao + ': y_test')
        plt.grid()
        plt.legend()
        plt.xlabel("Dias")
        plt.ylabel("Preços Padronizados")
        plt.show()

        #----------------------------------------------------
        # Configura rede LSTM
        #----------------------------------------------------
        trein_tam = x_trein.shape[0]
        #test_tam = x_test.shape[0]
        tf.compat.v1.reset_default_graph()
        X = tf.compat.v1.placeholder(tf.float32, [None, passosequenc, numentradas])
        y = tf.compat.v1.placeholder(tf.float32, [None, numsaidas])
        #indsaida = 0
        matrizrandomiz = np.arange(x_trein.shape[0])
        np.random.shuffle(matrizrandomiz)

        camadasrede = [tf.contrib.rnn.BasicLSTMCell(num_units=numneurons, activation=tf.nn.elu) 
                    for layer in range(numcamadas)]

        neuronmulticam = tf.contrib.rnn.MultiRNNCell(camadasrede)
        saidasrnn, estados = tf.nn.dynamic_rnn(neuronmulticam, X, dtype=tf.float32)
        saidasrnnempilh = tf.reshape(saidasrnn, [-1, numneurons]) 
        saidasempilh = tf.layers.dense(saidasrnnempilh, numsaidas)
        saidas = tf.reshape(saidasempilh, [-1, passosequenc, numsaidas])
        saidas = saidas[:,passosequenc-1,:] 

        # Função de Custo
        perda = tf.reduce_mean(tf.square(saidas - y))

        
        print(" ")
        print("saidas: ")
        print(saidas[:,:])
        print("y: ")
        print(y)
        print("perda: ")
        print(perda)
        

        # Otimizador ADAM
        otimizad = tf.compat.v1.train.AdamOptimizer(learning_rate=txaprendizado) 
        treinamtoproc = otimizad.minimize(perda)

        #----------------------------------------------------
        # Treina rede LSTM
        #----------------------------------------------------
        with tf.compat.v1.Session() as sesstrein: 
            sesstrein.run(tf.compat.v1.global_variables_initializer())
            for linhadados in range(int(numepocas*trein_tam/tamlote)):
                lote_x, lote_y = objeto.proxlote(tamlote) # fetch the next training batch 
                sesstrein.run(treinamtoproc, feed_dict={X: lote_x, y: lote_y}) 
                if linhadados % int(trein_tam/tamlote) == 0:
                    trein_errquadmed = perda.eval(feed_dict={X: x_trein, y: y_trein}) 
                    valid_errquadmed = perda.eval(feed_dict={X: x_valid, y: y_valid}) 
                    print('Iteracao: %4.0f Epoca: %4.0f EQM: treino %.6f validacao %.6f'%(
                            linhadados, linhadados*tamlote/trein_tam, trein_errquadmed, valid_errquadmed))
        
            # Resultado Predições
            testpredicao_y = sesstrein.run(saidas, feed_dict={X: x_test})
            
        print("")
        print("Mostra testpredicao")

        print("testpredicao_y[0:10,0]")
        print(testpredicao_y[0:10,0])

        print("testpredicao_y[0:10,1]")
        print(testpredicao_y[0:10,1])

        print("testpredicao_y[0:10,2]")
        print(testpredicao_y[0:10,2])

        print("testpredicao_y[0:10,3]")
        print(testpredicao_y[0:10,3])

        # Verificando predicao 
        testpredicao_y.shape

        #----------------------------------------------------
        # Grafico Alvo x Predicao
        #----------------------------------------------------
        comp = pd.DataFrame({'Col1':y_test[:,a],'Col2':testpredicao_y[:,a]})
        plt.figure(figsize=(10,5))
        plt.plot(comp['Col1'], color='blue', label='Alvo')
        plt.plot(comp['Col2'], color='red', label='Predição')
        plt.title(acao + ': Alvo x Predição')
        plt.grid()
        plt.legend()
        plt.xlabel("Dias")
        plt.ylabel("Preços Padronizados")
        plt.show()

        #----------------------------------------------------
        # Grafico Acuracia (Alvo - Predicao)
        #----------------------------------------------------
        comp = pd.DataFrame({'Col':testpredicao_y[:,a] - y_test[:,a]})
        plt.figure(figsize=(10,5))
        plt.plot(comp['Col'], color='black', label='Diferença entre Alvo e Predição')
        plt.title(acao + ': Acurácia (Alvo - Predição)')
        plt.grid()
        plt.legend()
        plt.xlabel("Dias")
        plt.ylabel("Diferença de Preços Padronizados")
        plt.show()

        #----------------------------------------------------
        # Media, Variancia e Desvio Padrao da Acurácia
        #----------------------------------------------------
        acur_mean = np.mean(comp)
        acur_var = np.var(comp)
        acur_std = np.std(comp)
        acur_correl = np.corrcoef(y_test[:,a],testpredicao_y[:,a])[0,1]
        acur_r2 = np.power(acur_correl,2)

        print('média         : %.6f ' %(acur_mean)) 
        print('variância     : %.6f ' %(acur_var)) 
        print('desvio padrão : %.6f ' %(acur_std))
        print('coef. correl. : %.6f ' %(acur_correl))
        print('coef. R2      : %.6f ' %(acur_r2))
        
        return np.around(acur_mean,6) , np.around(acur_var,6), np.around(acur_std,6), np.around(acur_r2,6)

    def teste(self, param1=0, param2=0):
        if param1 == 1:
            retorna = 'um'
        else:
            retorna = 'outro'
        
        return retorna

objeto = RedeLstm()
#objeto.treinarede()




