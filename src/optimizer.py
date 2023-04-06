import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import Normalize
from ax import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
import torch.nn as nn
import logging


class CNNTrainer:
    def __init__(self, train_data, val_data=None, batch_size=64, model=None, patience=5):
        self.patience = patience
        self.model = model
        # Load the training data
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
        if val_data: 
            self.val_loader = data.DataLoader(train_data, batch_size=batch_size)
            
    def compute_input_size(self, params):
        # Add BatchNorm2d layer to standardize input data
        x = nn.BatchNorm2d(3)(torch.zeros([1,3,32,32]))
        pool1 = eval(params.get("pool1"))
        pool2 = eval(params.get("pool2"))        
        # Pass the input through the sequence of layers
        for layer in [
                nn.Conv2d(in_channels=3, 
                          out_channels=params.get('num_filters1'), 
                          kernel_size=params.get('filter_size1')),
                nn.ReLU(),
                pool1(kernel_size=params.get("kernel_pool1")),
                nn.Conv2d(in_channels=params.get('num_filters1'),
                          out_channels=params.get('num_filters2'),
                          kernel_size=params.get('filter_size2')),
                nn.ReLU(),
                pool2(kernel_size=params.get("kernel_pool2")),
                nn.Conv2d(in_channels=params.get('num_filters2'),
                          out_channels=params.get('num_filters3'),
                          kernel_size=params.get('filter_size3')),
                nn.ReLU(),
                nn.Sequential(
                    nn.Conv2d(params.get('num_filters3'),
                              params.get('num_filters3'), 
                              kernel_size=1),
                    nn.Sigmoid()),            
                nn.Flatten()]:
            x = layer(x)
        input_size = x.shape[0] * x.shape[1]
        return input_size

    def build_model(self, params):
        linear_input = self.compute_input_size(params) 
        # Define the CNN architecture based on the given parameters
        pool1 = eval(params.get("pool1"))
        pool2 = eval(params.get("pool2"))
        print("linear_input", linear_input)
        model = nn.Sequential(
            nn.BatchNorm2d(3),  # Add BatchNorm2d layer to standardize input data,
            nn.Conv2d(in_channels=3, 
                      out_channels=params.get('num_filters1'), 
                      kernel_size=params.get('filter_size1')),
            nn.ReLU(),
            pool1(kernel_size=params.get("kernel_pool1")),
            nn.Conv2d(in_channels=params.get('num_filters1'),
                      out_channels=params.get('num_filters2'),
                      kernel_size=params.get('filter_size2')),
            nn.ReLU(),
            pool2(kernel_size=params.get("kernel_pool2")),
            nn.Conv2d(in_channels=params.get('num_filters2'),
                      out_channels=params.get('num_filters3'),
                      kernel_size=params.get('filter_size3')),
            nn.ReLU(),
           # Define the attention mechanism
            nn.Sequential(
                nn.Conv2d(params.get('num_filters3'),
                          params.get('num_filters3'), 
                          kernel_size=1),
                nn.Sigmoid()),
            nn.Flatten(),
            nn.Linear(linear_input, 10)
        )
        return model
        
        
    def fit(self, model, lr=0.001, epochs=1):
        # Initialize weights with Xavier initialization
                
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)      
                
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Set the number of threads for multi-threading
        torch.set_num_threads(8)
        
        # Initialize some variables for early stopping
        best_loss = float('inf')
        counter = 0  # Counter for number of epochs without improvement
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            print("Running epoch ", epoch)
            total, correct = 0,0
            for images, labels in self.train_loader:
                images, labels = images.to(device='cpu'), labels.to(device='cpu')
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Validate your model
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for x, y in self.val_loader:
                    y_pred = model(x)
                    val_loss += criterion(y_pred, y).item()
                val_loss /= len(self.val_loader)
            print('Val loss: ', val_loss)
            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print(f"Early stopping on epoch {epoch}")
                    break
        return model
        
class CNNPredictor:
    def __init__(self, model):
        self.model = model
        
    def predict(self, test_data):
        # Evaluate the model on the validation set
        data_loader = data.DataLoader(test_data, batch_size=64)
        
        # Calculate validation accuracy
        correct, total  = 0, 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return labels, accuracy


class CNNOptimizer:
    def __init__(self, search_space, train_data, val_data, steps=20, epochs=1):
        self.epochs = epochs
        self.steps = steps
        self.search_space = search_space
        self.train_data = train_data
        self.val_data = val_data

    def evaluate_model(self, parameterization):
        batch_size = parameterization.get("batch_size")
        lr = parameterization.get("lr")
        
        try:
            print("Testing config", parameterization)
            trainer = CNNTrainer(self.train_data, batch_size=batch_size)
            model = trainer.build_model(parameterization) 
        except Exception as e:
            logging.error(e)
            return {'acc': 0} 
        
        print("CONFIG Valida")
        model = trainer.fit(model, lr=lr, epochs=self.epochs)
        predictor = CNNPredictor(model)
        _, accuracy = predictor.predict(self.val_data)
        print("ACC during eval", accuracy)
        # Return the validation accuracy as the objective value to optimize
        return {'acc': accuracy}

    def optimize(self):
        
        constraints = ["num_filters1 <= num_filters2",    
                       "num_filters2 <= num_filters3",   
                       "filter_size1 >= filter_size2",   
                       "filter_size2 >= filter_size3"  
                      ]    

        best_parameters, best_values, experiment, model = optimize(
            parameters=self.search_space,
            evaluation_function=self.evaluate_model,
            parameter_constraints=constraints,
            objective_name='acc',
            minimize=False,
            total_trials=self.steps
        )

        print('Best parameters:', best_parameters)
        print('Best validation accuracy:', best_values[0])
        
        return best_parameters, best_values, experiment, model