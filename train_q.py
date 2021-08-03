from players.q_players.q_trainer import QTrainer
from players.q_players.mc_every_visit_qlearner.qlearner import QLearner
from players.q_players.mc_every_visit_deep_q.deep_qlearner import DeepQLearner
from environment.go import black_stone, white_stone

if __name__ == "__main__":
    qlearner = DeepQLearner(board_size=5, epsilon_start=0.1)
    qtrainer = QTrainer(qlearner)
    qtrainer.train(60000)
    # qtrainer = QTrainer(qlearner, learner_stone=white_stone)
    # qtrainer.train(20)