from players.q_players.q_trainer import QTrainer

if __name__ == "__main__":
    qtrainer = QTrainer()

    qtrainer.train(100001)