class wallet:
    def __init__(self, usd):
        self.usd = usd
        self.crypto = 0
    
    def buy(self, crypto_am, price):
        self.crypto += crypto_am
        self.usd -= price*crypto_am

    def sell(self, crypto_am, price):
        self.crypto -= crypto_am
        self.usd += price*crypto_am

    def info(self):
        return self.usd, self.crypto
