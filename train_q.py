from players.q_players.q_trainer import QTrainer
from players.q_players.mc_every_visit_qlearner.qlearner import QLearner
from environment.go import black_stone, white_stone

if __name__ == "__main__":
    qlearner = QLearner()
    # qtrainer = QTrainer(qlearner)
    # qtrainer.train(20)
    qtrainer = QTrainer(qlearner, learner_stone=white_stone)
    qtrainer.train(20)