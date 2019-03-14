#code for NeuralNetwork classifier
class NeuralNetwork(nn.Module):
    #define layers of neural network
    def __init__(self, input_size, output_size, hidden_layers, dropout):
        super().__init__()
        #input size to hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        #add hidden layers
        i = 0
        j = len(hidden_layers)-1
            
        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1
        #check to make sure hidden layers are formatted correctly
        for each in hidden_layers:
            print(each)
        #last hidden layer to output
        self.output = nn.Linear(hidden_layers[j], output_size)
        self.dropout = nn.Dropout(p = dropout)
    #feed forward method
    def forward(self, tensor):
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
            tensor = self.dropout(tensor)
        tensor - self.output(tensor)
        #log softmax
        return F.log_softmax(tensor, dim=1)
