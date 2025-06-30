import torch
import numpy as np
from numpy.linalg import norm
import pickle


class Memory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size, dimension=1 * 3 * 22 * 22, alpha=0.9): #1 * 3 * 224 * 224
        self.memory = {}
        self.size = size
        self.dimension = dimension
        self.alpha = alpha

    def reset(self):
        self.memory = {}

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        # mo = 0.5 
        keys = keys.reshape(len(keys), self.dimension)
        for i, key in enumerate(keys):
            
            if len(self.memory.keys()) >= self.size:
                # Memory is full, find the nearest neighbours and update them
                #key = key.reshape(len(key), self.dimension)
                all_keys = np.frombuffer(np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(self.get_size(), self.dimension)
                similarity_scores = np.dot(all_keys, key.T) / (norm(all_keys, axis=1) * norm(key.T))
                top_k_indices = np.argsort(similarity_scores)[-5:]  # Top-k indices with highest similarity
                for idx in top_k_indices:
                    mem_key = all_keys[idx].tobytes()
                    top_k_logit = self.memory[mem_key]
                    #self.memory[mem_key] = mo * top_k_logit + (1 - mo) * logits[i]
                    # Update the memory with a weighted average of the top-k logits
                    self.memory[mem_key] = self.alpha * top_k_logit + (1 - self.alpha) * logits[i]
                """
                neighbors, similarity_scores = self.get_topk(np.array([key_flat]), k=5)

                for nkey, score in zip(neighbors, similarity_scores):
                    mem_key = nkey.tobytes()
                    self.memory[mem_key] = self.alpha * self.memory[mem_key] + (1 - self.alpha) * logits[i]

                """
            else:
                # Memory is not full, add new key-logit pair to memory
                self.memory.update({key.reshape(self.dimension).tobytes(): logits[i]})    
        

    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []

        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)


    
    def save_memory(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, file_path):
        with open(file_path, 'rb') as f:
            self.memory = pickle.load(f)


    def get_topk_avg(self, keys, k):
        
        samples = []

        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            mean_prompts = torch.FloatTensor(np.mean(neighbours, axis=0))  # (dimension,)
            # batch = self._prepare_batch(neighbours, attention_weight) #TEST
            samples.append(mean_prompts)

            #attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            #batch = self._prepare_batch(neighbours, attention_weight)
            #samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)
    


#---------------------------------------------------------------------------------------Contextual Evolution Memory
class Memory_vft(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size, dimension=1 * 1536, key_dimension=1*768, alpha=0.1): #, key_dim=768, dim = 1536
        # self.memory = {}
        self.size = size
        self.dimension = dimension
        self.key_dimension = key_dimension
        self.alpha = alpha
        # self.dim = dim
        # self.key_dim = key_dim
        logits = torch.randn(size, dimension)
        #self.memory = {torch.randn(1, key_dimension): torch.randn(1, dimension) for _ in range(size)}
        self.memory = {
            torch.randn(1, key_dimension).numpy().tobytes(): torch.randn(1, dimension) 
            for _ in range(size)
        }
       
        

    def reset(self):
        self.memory = {}

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        
        keys = keys.reshape(len(keys), self.key_dimension)
        for i, key in enumerate(keys):
            
            if len(self.memory.keys()) >= self.size:
                # Memory is full, find the nearest neighbours and update them
                #key = key.reshape(len(key), self.dimension)
                all_keys = np.frombuffer(np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(self.get_size(), self.key_dimension)
                similarity_scores = np.dot(all_keys, key.T) / (norm(all_keys, axis=1) * norm(key.T))
                top_k_indices = np.argsort(similarity_scores)[-5:]  # Top-k indices with highest similarity
                for idx in top_k_indices:
                    mem_key = all_keys[idx].tobytes()
                    #mem_key = all_keys[idx].tobytes()
                    top_k_logit = self.memory[mem_key]
                    #self.memory[mem_key] = mo * top_k_logit + (1 - mo) * logits[i]
                    self.memory[mem_key] = self.alpha * top_k_logit + (1 - self.alpha) * logits[i]

            else:
                self.memory.update({key.reshape(self.key_dimension).tobytes(): logits[i]})

       
        

        
        

    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []

        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)

    def save_memory(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, file_path):
        with open(file_path, 'rb') as f:
            self.memory = pickle.load(f)


    def get_topk(self, keys, k):
        
        samples = []

        keys = keys.reshape(len(keys), self.key_dimension) #(num_keys, dimension)
        total_keys = len(self.memory.keys())

        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.key_dimension) #(total_keys, dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]] #(k, dimension)
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys] #(k,dimension)

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight) #TEST
            samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)
        # return samples, np.mean(similarity_scores)
    

    def get_topk_mean(self, keys, k):
        
        samples = []

        keys = keys.reshape(len(keys), self.dimension) #(num_keys, dimension)
        total_keys = len(self.memory.keys())

        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension) #(total_keys, dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]] #(k, dimension)
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys] #(k,dimension)

            # attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            # mean_prompts = np.mean(neighbours, axis=0)  # (dimension,)
            mean_prompts = torch.FloatTensor(np.mean(neighbours, axis=0))  # (dimension,)
            # batch = self._prepare_batch(neighbours, attention_weight) #TEST
            samples.append(mean_prompts)

        return torch.stack(samples), np.mean(similarity_scores)
    

    

    def retrieve_prompt_add_avg(self, avg_pano_embeds, combined, top_k=16): #now for usage of visual feature
        """
        Retrieve top-k similar prompts from memory for each directional pano_embeds (1*12*768)
        and prepend them to form new pano_embeds with prompts.

        Args:
            avg_pano_embeds: Tensor of shape (1, 768) representing the average panoramic embedding.
            top_k: Number of top similar prompts to retrieve from memory.

        Returns:
            avg_pano_with_prompts: Tensor of shape (1, 768) with enhanced embeddings.
        """
        ud = 0.2
        # Initialize a list to store the updated pano_embeds with prompts
        pano_with_prompts = []

        # Define a linear layer to project concatenated prompts to the desired dimension
        linear_layer = torch.nn.Linear(in_features=1536, out_features=768)

        # Iterate over each direction in pano_embeds (12 directions in total)
        if isinstance(avg_pano_embeds, torch.Tensor):
            avg_pano_embeds = avg_pano_embeds.detach().cpu().numpy()

        posprompts, _ = self.get_topk(keys=avg_pano_embeds, k=top_k)

                # Calculate the mean of the top-k prompts (1*768)
            #mean_prompt = torch.from_numpy(prompts).mean(dim=0, keepdim=True)
            #mean_prompt = prompts.mean(dim=0, keepdim=True)
        avg_pano_embeds = torch.from_numpy(avg_pano_embeds).float()
        combined_embeds = torch.from_numpy(combined).float()
                # Concatenate the mean_prompt with direction_embed
            #concatenated = torch.cat([mean_prompt, avg_pano_embeds], dim=-1)  # (1, 1536)

                # Use the linear layer to project back to (1, 768)
            #enhanced_embed = linear_layer(concatenated)
        enhanced_embed = combined_embeds * (1-ud) + posprompts.squeeze(0) * ud

                # Add the enhanced result to the pano_with_prompts list
            #pano_with_prompts.append(enhanced_embed)

            # Stack the updated embeddings back to form a tensor of shape (1, 12, 768)
        pano_with_posprompts = enhanced_embed

        return pano_with_posprompts
    
