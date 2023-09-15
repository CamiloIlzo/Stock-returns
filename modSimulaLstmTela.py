# -*- coding: utf-8 -*-
#"""
#Created on Sun Jun 14 19:27:18 2020
#
#@author: CIS-NB
#"""

import PySimpleGUI as sg
from modLstm import RedeLstm 

rl = RedeLstm()

#sg.theme('DarkBlack')
#sg.theme('DarkBlue2')

class TelaPython:
    def __init__(self):
        #sg.ChangeLookAndFeel() 
        # Layout
        layout = [
            [sg.Text('Rede LSTM aplicada ao Algorithmic Trading', size=(40,0), font='Arial 12' )],
            [sg.Text('Épocas', size=(10,0)), sg.Combo(default_value=5, values=[i for i in range(5,101,5)], key='epocas', size=(5,0), readonly=True ,disabled=False)],
            [sg.Text('Tx.Aprendiz', size=(10,0)), sg.Combo(default_value=0.001, values=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010], key='txaprend', size=(5,0), readonly=True, disabled=False)],
            [sg.Text('Ação', size=(10,0)), sg.Combo(default_value='PETR4', values=['PETR4', 'VALE3'], key='acao', size=(10,0), readonly=True, disabled=False)],
            #[sg.Text('Lote', size=(10,0)), sg.Combo(default_value=50, values=[i for i in range(10,101,10) ], key='lote', size=(5,0), readonly=True ,disabled=False)],
            #[sg.Text('Camadas', size=(10,0)), sg.Combo(default_value=2, values=[i for i in range(1,4)], key='camadas', size=(5,0), readonly=True, disabled=False)],
            #[sg.Text('Unidades', size=(10,0)), sg.Combo(default_value=200, values=[i for i in range(10,301,10)], key='unidades', size=(5,0), readonly=True, disabled=False)],
            #[sg.Text('Sequência', size=(10,0)), sg.Combo(default_value=20, values=[i for i in range(10,101,10)], key='sequencia', size=(5,0), readonly=True, disabled=False)],
            #[sg.Slider(range=(0,100), default_value=0, orientation='h', size=(15,20),key='sliderVeloc')],
            [sg.Text('Média         ', size=(15,0), text_color = 'black'), sg.Text('0.00', key='media'       , size=(10,0), text_color = 'black')],
            [sg.Text('Variância     ', size=(15,0), text_color = 'black'), sg.Text('0.00', key='variancia'   , size=(10,0), text_color = 'black')],
            [sg.Text('Desvio Padrão ', size=(15,0), text_color = 'black'), sg.Text('0.00', key='desviopadrao', size=(10,0), text_color = 'black')],
            [sg.Text('Coef. Det. R2 ', size=(15,0), text_color = 'black'), sg.Text('0.00', key='coefdet'     , size=(10,0), text_color = 'black')],
            #[sg.Button('Ok')],
            #[sg.Button('Cancel')],
            #[sg.Output(size=(30,20))
            #[sg.Text('Arquivo de Dados')],
            #[sg.Input(), sg.FileBrowse(file_types=(("ALL Files", "*.csv"),),)],
            #[sg.Button('Execute', button_color=('white', 'black'))],
            [sg.Button('Execute')],
            #[sg.Cancel(button_color=('white', 'black'))]
            [sg.Cancel()]
            ]
        
        # Janela
        self.janela = sg.Window("Deep Learning - LSTM").Layout(layout)
        
    def Iniciar(self):
        while True:
            #Extrair dados da Tela
            
            self.evento, self.values = self.janela.Read()
            epocas = self.values['epocas']
            txaprend = self.values['txaprend']
            acao = self.values['acao']
            if self.evento in (sg.WINDOW_CLOSED,'Cancel'):
                break
            else:
                print(self.values)
                acurmean, acurvar, acurstd, acurr2 = rl.treinarede(epocas,txaprend, acao)
                #print('')
                #print('média         : %.6f ' %(acurmedia))
                self.janela['media'].Update(float(acurmean))
                self.janela['variancia'].Update(float(acurvar))
                self.janela['desviopadrao'].Update(float(acurstd))
                self.janela['coefdet'].Update(float(acurr2))
                #self.janela.Refresh() 
                
                #resultado = rl.teste(1,2) 
                #print('')
                #print(resultado)

            #Tratamento de eventos da tela
            #elif self.evento in ('OK', 'Ok'):
                #dlat.lstm_executa
                        
            ## change the "output" element to be the value of "input" element
            #window['-OUTPUT-'].update(values['-IN-'])

        self.janela.close(); del self.janela

def main():
    tela = TelaPython()
    tela.Iniciar()

if __name__ == "__main__":
    main()
