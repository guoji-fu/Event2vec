import numpy as np
import tensorflow as tf

class EVENT2VEC(object):
    def __init__(self,
                 nodes_num,
                 node_types,
                 nodes_ind,
                 beta=30,
                 rep_size=128,
                 struct=[None, None],
                 epochs=2000,
                 batch_size=128,
                 learning_rate=0.01):
        self.nodes_num = nodes_num
        self.node_types = node_types
        self.nodes_ind = nodes_ind
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.input_dim = nodes_num
        self.nodes_hidden_dim = rep_size
        self.events_hidden_dim = rep_size

        self.embeddings = None
        self.vectors = {}
        self.W = {}
        self.b = {}

        self.layers = len(struct)  

        self.struct = {}

        self.inc_mat = {}

        self.node_weight = {}

        for type_i in range(len(self.node_types)):
            node_type = self.node_types[type_i]
            self.struct[node_type] = struct
            self.struct[node_type][0] = self.input_dim[node_type]
            self.struct[node_type][-1] = self.nodes_hidden_dim

            self.inc_mat[node_type] = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim[node_type]])

            node_struct = self.struct[node_type]
            node_encoded = {}

            for i in range(self.layers-1):
                name = node_type + 'encoder' + str(i)
                self.W[name] = tf.Variable(tf.random_normal([node_struct[i], node_struct[i+1]], dtype=tf.float32), name=name)
                self.b[name] = tf.Variable(tf.zeros([node_struct[i+1]], dtype=tf.float32), name=name)
            
            node_struct.reverse()
            for i in range(self.layers-1):
                name = node_type + 'decoder' + str(i)
                self.W[name] = tf.Variable(tf.random_normal([node_struct[i], node_struct[i+1]], dtype=tf.float32), name=name)
                self.b[name] = tf.Variable(tf.zeros([node_struct[i+1]], dtype=tf.float32), name=name)

        node_encoded = self.nodes_encoder(self.inc_mat)
        event_encoded = self.event_encoder(node_encoded)
        decoded = self.decoder(event_encoded)

        self.node_encoded = node_encoded
        self.event_encoded = event_encoded
        self.decoded = decoded

        self.loss = self.all_loss()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()
    
    def nodes_encoder(self, X):
        X_encoded = {}
        for type_i in range(len(self.node_types)):
            node_type = self.node_types[type_i]
            X_encoded[node_type] = X[node_type]
            for i in range(self.layers-1):
                name = node_type + 'encoder' + str(i)
                X_encoded[node_type] = tf.nn.sigmoid(tf.matmul(X_encoded[node_type], self.W[name]) + self.b[name])
        
        return X_encoded

    def event_encoder(self, X):
        event_encoded = None
        for type_i in range(len(self.node_types)):
            node_type = self.node_types[type_i]
            if event_encoded is None:
                event_encoded = X[node_type]
            else:
                event_encoded += X[node_type]

        return event_encoded
    
    def decoder(self, X):
        X_decoded = {}
        for type_i in range(len(self.node_types)):
            node_type = self.node_types[type_i]
            X_decoded[node_type] = X
            for i in range(self.layers-1):
                name = node_type + 'decoder' + str(i)
                X_decoded[node_type] = tf.nn.sigmoid(tf.matmul(X_decoded[node_type], self.W[name]) + self.b[name])

        return X_decoded

    def all_loss(self):
        def get_2nd_loss(X, newX, beta):
            loss = None
            B = {}
            for node_type in self.node_types:
                B[node_type] = X[node_type] * (beta - 1) + 1
                if loss is None:
                    loss = tf.reduce_sum(tf.pow(tf.subtract(X[node_type], newX[node_type])*B[node_type], 2))                 
                else:
                    loss += tf.reduce_sum(tf.pow(tf.subtract(X[node_type], newX[node_type])*B[node_type], 2))
            
            return loss

        def get_reg_loss(weights, biases):
            ret1 = 0
            ret2 = 0
            for type_i in range(len(self.node_types)):
                node_type = self.node_types[type_i]
                for i in range(self.layers-1):
                    name1 = node_type + 'encoder' + str(i)
                    name3 = node_type + 'decoder' + str(i)
                    ret1 = ret1 + tf.nn.l2_loss(weights[name1]) + tf.nn.l2_loss(weights[name3])
                    ret2 = ret2 + tf.nn.l2_loss(biases[name1]) + tf.nn.l2_loss(biases[name3])

            ret = ret1 + ret2

            return ret
        
        def get_loss_xxx(X):
            loss = None
            for node_type in self.node_types:
                if loss is None:
                    loss = tf.reduce_sum(tf.pow(X[node_type], 2))
                else:
                    loss += tf.reduce_sum(tf.pow(X[node_type], 2))
            
            return loss

        self.loss_2nd = get_2nd_loss(self.inc_mat, self.decoded, self.beta)
        self.loss_reg = get_reg_loss(self.W, self.b)
        return self.loss_2nd + self.loss_reg

    def get_batch(self, X, batch_size):
        a = np.random.choice(len(X[self.node_types[0]]), batch_size, replace=False)
        batch_data = {}
        for type_i in range(len(self.node_types)):
            node_type = self.node_types[type_i]
            batch_data[node_type] = X[node_type][a]
        
        return batch_data
    
    def get_feed_dict(self, batch_data):
        feed_dict = {}
        for type_i in range(len(self.node_types)):
            node_type = self.node_types[type_i]
            feed_dict[self.inc_mat[node_type]] = batch_data[node_type]

        return feed_dict

    def train(self, inc_mat):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epochs):
                embeddings = None
                for j in range(np.shape(inc_mat[self.node_types[0]])[0] // self.batch_size):
                    batch_data = self.get_batch(inc_mat, self.batch_size)
                    loss, _ = sess.run([self.loss, self.train_op], feed_dict=self.get_feed_dict(batch_data))
                    if embeddings is None:
                        embeddings = self.node_encoded
                    else:
                        for type_i in range(len(self.node_types)):
                            node_type = self.node_types[type_i]
                            embeddings[node_type] = np.vstack((embeddings[node_type], self.node_encoded))
                    # print('batch {0}: loss = {1}'.format(j, loss))
                print('epoch {0}: loss = {1}'.format(i, loss))
            self.saver.save(sess, './model.ckpt')
    
    def get_embeddings(self, inc_mat, node_deg):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './model.ckpt')
            look_back = self.nodes_ind
            vectors = {}
            data = self.get_feed_dict(inc_mat)
            event_embeddings = sess.run(self.event_encoded, feed_dict=data)
            for node_type in self.node_types:
                node_embeddings = np.dot(np.linalg.inv(node_deg[node_type]), np.dot(inc_mat[node_type].T, event_embeddings))
                for i, embedding in enumerate(node_embeddings):
                    vectors[look_back[node_type][i]] = embedding
                
        return vectors

    def save_embeddings(self, filename, inc_mat, node_deg):
        # print(inc_mat)
        self.vectors = self.get_embeddings(inc_mat, node_deg)
        node_size = 0
        for node_type in self.node_types:
            node_size += len(inc_mat[node_type].T)
        fout = open(filename, 'w')
        fout.write('{} {}\n'.format(node_size, self.nodes_hidden_dim))
        for node, vec in self.vectors.items():
            fout.write('{} {}\n'.format(str(node), ' '.join([str(x) for x in vec])))
        
        fout.close()



