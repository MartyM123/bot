class wallet:
    def __init__(self, start_budget):
        self.start_budget = start_budget
        self.usd = start_budget
        self.crypto = 0
        self.transactions = []


    def buy(self, crypto_am, price):
        self.crypto += crypto_am
        self.usd -= price*crypto_am
        self.transactions.append('buy')

    def sell(self, crypto_am, price):
        self.crypto -= crypto_am
        self.usd += price*crypto_am
        self.transactions.append('sell')

    def info(self):
        return self.usd, self.crypto

    def sell_all(self, price):
        self.sell(crypto_am=self.crypto, price=price)

    def wait(self):
        self.transactions.append('wait')
