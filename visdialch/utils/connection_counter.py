

class ConnectionCounter(object):
    def __init__(self):
        self.n_connct = 0

    def add(self, mask):
    	# mask to the number
    	new_connct = 0
        self.n_connct += new_connct

    def retrieve(self):
        return self.n_connct    

    def reset(self):
        self.n_connct = 0        
