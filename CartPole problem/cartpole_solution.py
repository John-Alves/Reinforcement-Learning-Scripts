# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importa o GYM e inicia o ambiente do CartPole 
import gym
env = gym.make('CartPole-v0')
env.env.theta_threshold_radians = np.radians(12) # Muda o limite do angulo

# Define o estado baseado nas variaveis do ambiente
def obter_estado(observacao):
    
    x = observacao[0]
    x_vel = observacao[1]
    ang = observacao[2]
    ang_vel = observacao[3]
    
    estado = 0
        
    # Discritizando posição em 0.4 unidades
    if  x < -2:    estado += 0
    elif x < -1.6: estado += 1
    elif x < -1.2: estado += 2
    elif x < -0.8: estado += 3
    elif x < -0.4: estado += 4
    elif x < -0:   estado += 5
    elif x < 0.4:  estado += 6
    elif x < 0.8:  estado += 7
    elif x < 1.2:  estado += 8
    elif x < 1.6:  estado += 9
    else:          estado += 10
    
    # Discretizando a velocidade
    if  x_vel < -0.5:   estado += 0
    elif x_vel < 0:     estado += 11
    elif x_vel < 0.5:   estado += 22
    else:               estado += 33
    
    # Discretizando o angulo
    if ang < np.radians(-8):   estado += 0
    elif ang < np.radians(-6): estado += 44
    elif ang < np.radians(-4): estado += 88
    elif ang < np.radians(-2): estado += 132
    elif ang < np.radians(0):  estado += 176
    elif ang < np.radians(2):  estado += 220
    elif ang < np.radians(4):  estado += 264
    elif ang < np.radians(6):  estado += 308
    elif ang < np.radians(8):  estado += 352
    else:                      estado += 396
    
    # Discretizando a velocidade angular
    if  ang_vel < np.radians(-20):    estado += 0
    elif ang_vel < np.radians(20):    estado += 440
    else:    			          estado += 880
        
    return estado

def obter_acao(tabela_Q, epson, estado):
    # Faz com que vez e outra sejam tomadas decisoes aleatorias
    if np.random.rand() <= epson:
        return env.action_space.sample()
    else: # Utiliza uma politica gananciosa (greedy)
        # Retorna a ação que possui a maior recompensa
        if tabela_Q[estado][0] > tabela_Q[estado][1]:
            return 0
        elif tabela_Q[estado][0] < tabela_Q[estado][1]:
            return 1
        else:
            return env.action_space.sample()
            

def atualiza_Q(tabela_Q, acao, estado, R, obs, alfa, gama):
    '''
    Q(s,a) = Q(s,a) + alfa * (R + gama * Q(s', a') - Q(s,a))
    '''
    # Calcula o valor de Q(s',a')
    n_estado = obter_estado(obs)
    valor_fut = tabela_Q[n_estado][acao]
    # Calcula o novo Q(s,a)
    tabela_Q[estado][acao] += alfa * (R + (gama * valor_fut) - tabela_Q[estado][acao])
    return tabela_Q

# Inicializa a tabela Q com zeros
tabela_Q = np.zeros((1760,2))
#tabela_Q = np.zeros((440,2))
# Define um perc de vezes no qual o agente irá tomar acoes aleatorias
epson = 0.05
# Define a taxa de aprendizado
alfa = 0.05
# Define o fator de desconto
gama = 0.99

# Executa os episodios atualizando a tabela Q
steps_res = []
total_epi = 1000
for i in range(total_epi):
    # Reseta o ambiente (volta para o estado inicial)
    obs = env.reset()
    # Diminui o valor de epson
    epson = epson/1.0005
    # Executa os movimentos
    for t in range(240):
        # Renderiza o ambiente
        #env.render()
        # Descobre qual o estado atual baseado nas observações
        estado = obter_estado(obs)    
        # Escolhe uma ação (usando e-greedy)
        acao = obter_acao(tabela_Q, epson, estado)
        # Executa a ação e recebe novas inf do ambiente
        obs, R, done, info = env.step(acao)
        # Atualiza a tabela Q
        tabela_Q = atualiza_Q(tabela_Q, acao, estado, R, obs, alfa, gama)
        # Fim do episodio
        if done:
            steps_res.append(t+1)
            print("Fim do Episodio {epi}. {step} movimentos executados.".format(step=t+1, epi=i))
            break
        
env.close()

# exporta a tabela Q para um arquivo CSV
#df = pd.DataFrame(tabela_Q, columns=["Esq", "Dir"])
#df.to_csv("tabela{var}.csv".format(var=i), columns=["Esq", "Dir"], index=False, sep=";")

# Plota um gráfico dos resultados
plt.plot(range(total_epi), steps_res)