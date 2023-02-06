import numpy as np
import random
import pickle

# Entorno del juego de tres palos
class Environment:
    def __init__(self):
        self.state = 0 #Se inicia el entorno con la posicion del agente en 0

    def reset(self):
        self.state = 0 #Cuando se reinicie el entorno, empieza con la posicion del agente en 0 de vuelta
        return self.state

    def step(self, action):
        if action == 0: #Si la accion es 0, se va a mover a la izquierda
            self.state = max(0, self.state - 1) #Si action es 0 se supone que el agente quiere moverse a la izquierda,
                                                #asi que se actualiza su posicion (state). Con la función max(), 
                                                #nos aseguramos que no salga de nuestro entorno, o sea, que no llegue
                                                #a ser un numero negativo
                                                #La funcion max retorna el numero mayor entre 0 y la posicion del agente
                                                #obligando a que si está en 0 no pueda seguir avanzando a la izq
        else: # Mover hacia la derecha
            self.state = min(2, self.state + 1) #Lo contrario al caso anterior, que no se pase de dos, y que vaya para 
                                                #la derecha. La funcion min retorna el minimo entre 2 y la posición actual
                                                #obligando al agente a que no se vaya del limite de la derecha

        # Calcular la recompensa
        reward = 0
        if self.state == 2: #Si el agente llegó hasta la derecha del todo, se lo recompensa
            reward = 1 #La recompensa es una forma de dar retroalimentación al agente 
                       #sobre la calidad de su acción, permitiéndole saber si está avanzando en la dirección 
                       #correcta para alcanzar su objetivo.

        return self.state, reward #devuelve la ubicacion del agente, y si obtuvo o no recompensa
        

# Agente que aprende por refuerzo
class Agent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states #El numero total de posiciones (se declara más abajo)
        self.num_actions = num_actions #El numero total de acciones
        self.Q = np.zeros((num_states, num_actions)) #Se crea la tabla Q-values y se llena con 0. La tabla va a tener los estados y acciones que se indiquen mas adelante

    def choose_action(self, state, epsilon): #epsilon es la certeza del agente. Cuanto menor sea, más se va a guiar de la Q-table. Cuanto mayor sea, más va a elegir valores al azar
        if np.random.uniform(0, 1) < epsilon: #Si el numero que se genera es menor a epsilon (0.1), se elije una decisión al azar (Hay un 10% de chances de que pase)
            action = random.choice([0, 1]) # Tomar una acción al azar
        else:
            action = np.argmax(self.Q[state, :]) # Tomar la acción con la mayor Q-value (90% de chance)
        return action

    def update_Q(self, state, action, reward, next_state):
        # Alpha representa la tasa de aprendizaje, es decir, cuánto de la nueva información se aplica a la actualización 
        # de la Q-table. Un valor pequeño significa que la Q-table se actualiza lentamente, permitiendo al agente 
        # aprender gradualmente a través de la experiencia. Un valor grande significa que la Q-table se actualiza 
        # rápidamente, permitiendo al agente aprender rápidamente, pero también puede resultar en overfitting.
        alpha = 0.1
        gamma = 0.99
        self.Q[state, action] = (1 - alpha) * self.Q[state, action] + alpha * (reward + gamma * np.max(self.Q[next_state, :]))


# Entrenar al agente
def train(agent, env, num_episodes):
    for episode in range(num_episodes): #para cada episodio
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, epsilon=0.1)
            print(action)
            next_state, reward = env.step(action)
            agent.update_Q(state, action, reward, next_state)
            state = next_state
            if reward == 1:
                done = True


# Inicializar el entorno y el agente
env = Environment()
agent = Agent(num_states=3, num_actions=2)


# Entrenar al agente
train(agent, env, num_episodes=1000)

# Ver la Q-table resultante
print(agent.Q)
