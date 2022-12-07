import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import os, json, copy, pickle
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value
from argparse import ArgumentParser
from collections import deque
from tqdm import tqdm
import em as dnd
import controller

USE_CUDA = torch.cuda.is_available()
# USE_CUDA= False
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)





class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, h_trj, action, reward, old_reward, next_state, nh_trj, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        h_trj = np.expand_dims(h_trj, 0)
        nh_trj = np.expand_dims(nh_trj, 0)

        self.buffer.append((state, h_trj, action, reward, old_reward, next_state, nh_trj, done))

    def sample(self, batch_size):
        state, h_trj, action, reward, old_reward, next_state, nh_trj, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.concatenate(h_trj), action, reward, old_reward, \
        np.concatenate(next_state), np.concatenate(nh_trj), done

    def __len__(self):
        return len(self.buffer)




mse_criterion = nn.MSELoss()


# plt.plot([epsilon_by_frame(i) for i in range(10000)])
# plt.show()
def inverse_distance(h, h_i, epsilon=1e-3):
    return 1 / (torch.dist(h, h_i) + epsilon)

def gauss_kernel(h, h_i, w=0.5):
    return torch.exp(-torch.dist(h, h_i)**2/w)

def no_distance(h, h_i, epsilon=1e-3):
    return 1


cos = nn.CosineSimilarity(dim=0, eps=1e-6)
def cosine_distance(h, h_i):
    return max(cos(h, h_i),0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class DQN_DTM(nn.Module):
    def __init__(self, env, args):
        super(DQN_DTM, self).__init__()
        if "maze_img" in args.task2:
            if "cnn" not in args.task2:
                self.num_inputs = 8
                self.proj= nn.Linear(514, 8)
            else:
                self.cnn = img_featurize.CNN(64)
                self.num_inputs = 64

        elif "world" in args.task2:
            self.cnn = img_featurize.CNN2(256)
            self.num_inputs = 256
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            self.num_inputs = env.observation_space.shape[0]
        if "trap" in args.task2:
            self.num_inputs = self.num_inputs + 2
        self.num_actions = env.action_space.n
        self.model_name=args.model_name
        self.num_warm_up=args.num_warm_up
        self.replay_buffer = args.replay_buffer
        self.gamma = args.gamma
        self.last_inserts=[]
        self.insert_size = args.insert_size
        self.args= args

        self.qnet = nn.Sequential(
            nn.Linear(self.num_inputs, args.qnet_size),
            nn.ReLU(),
            nn.Linear(args.qnet_size, args.qnet_size),
            nn.ReLU(),
            nn.Linear(args.qnet_size, env.action_space.n)
        )

        self.qnet_target = nn.Sequential(
            nn.Linear(self.num_inputs, args.qnet_size),
            nn.ReLU(),
            nn.Linear(args.qnet_size, args.qnet_size),
            nn.ReLU(),
            nn.Linear(args.qnet_size, env.action_space.n)
        )


        if args.write_interval<0:
            self.state2key = nn.Linear(self.num_inputs, args.hidden_size)
            self.dnd = dnd.DND(no_distance, num_neighbors=args.k, max_memory=args.memory_size, lr=args.write_lr)
        else:
            self.dnd = dnd.DND(inverse_distance, num_neighbors=args.k, max_memory=args.memory_size, lr=args.write_lr)

        self.emb_index2count = {}
        self.act_net = nn.Linear(self.num_actions, self.num_actions)
        self.act_net_target = nn.Linear(self.num_actions, self.num_actions)

        self.choice_net = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),nn.ReLU(),
            nn.Linear(args.hidden_size, 1))
        self.choice_net_target = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),nn.ReLU(),
            nn.Linear(args.hidden_size, 1))
        self.alpha = nn.Parameter(torch.tensor(1.0),
                                  requires_grad=True)
        self.alpha_target = nn.Parameter(torch.tensor(1.0),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.ones(self.num_actions),
                                  requires_grad=True)
        self.trj_model = controller.LSTMController(self.num_inputs+self.num_actions, args.hidden_size, num_layers=1)
        self.trj_out = nn.Linear(args.hidden_size, self.num_inputs+self.num_actions+1)
        self.reward_model = nn.Sequential(
            nn.Linear(self.num_inputs+self.num_actions+args.hidden_size, args.reward_hidden_size),
            nn.ReLU(),
            nn.Linear(args.reward_hidden_size,args.reward_hidden_size),
            nn.ReLU(),
            nn.Linear(args.reward_hidden_size, 1),
        )
        self.best_trj = []
        self.optimizer_dnd = torch.optim.Adam(self.trj_model.parameters())
        # self.future_net = nn.Sequential(
        #     nn.Linear(args.hidden_size, args.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(args.hidden_size, args.hidden_size),
        # )
        self.future_net = nn.Sequential(
            nn.Linear(args.hidden_size, args.mem_dim),
        )

        for param in self.future_net.parameters():
            param.requires_grad = False
        if args.rec_period<0:
            for param in self.trj_model.parameters():
                param.requires_grad = False

        self.apply(weights_init)

    def forward(self, x, h_trj, episode=0, use_mem=1.0, target=0, r=None, a=None, is_learning=False):

        q_value_semantic = self.semantic_net(x, target)
        z = torch.zeros(x.shape[0], self.num_actions)
        if USE_CUDA:
            z = z.cuda()
        q_episodic = qt= z
        if episode > self.num_warm_up and  random.random() <use_mem and self.dnd.get_mem_size()>1:
            plan_step = 1
            lx = x
            lh_trj = h_trj
            a0=a

            if self.args.write_interval<0:
                fh_trj = self.state2key(x)
                q_episodic = self.dnd.lookup(fh_trj, is_learning=is_learning, p=args.pread)
                #print(q_episodic)
                # print("q lookup", q_estimates)
            else:
                if a is not None:
                    for i in range(plan_step):
                        lx, h_trj_a = self.trj_model(self.make_trj_input(lx, a, r), lh_trj)
                        if plan_step>1 or r is not None:
                            lx = self.trj_out(lx)
                        lh_trj = h_trj_a
                        if len(lx.shape)>1:
                             lx = lx[:,:self.num_inputs]
                        else:
                            lx = lx[:self.num_inputs]

                        if r is None:
                            r = self.reward_model(
                                torch.cat([self.make_trj_input(lx, a), lh_trj[0][0].detach()], dim=-1))

                        q_episodic[:, a0] = r + self.args.gamma*self.episodic_net(h_trj_a, is_learning)
                        if plan_step>1:
                            for aa in range(self.num_actions):
                                _, h_trj_aa = self.trj_model(self.make_trj_input(lx, aa, r), h_trj_a)
                                qt[:, aa] = self.episodic_net(h_trj_aa, is_learning)
                            a= qt.max(1)[1]

                else:
                    # if random.random()<0.1:
                    #     if self.best_trj:
                    #         q_episodic = self.exploit(lx)
                    # else:
                    #print("plan")
                    for a in range(self.num_actions):
                        #print(a)
                        # print(self.make_trj_input(lx, a))
                        lxx, h_trj_aa = self.trj_model(self.make_trj_input(lx, a), lh_trj)
                        if r is  None:
                            # lxx = self.trj_out(lxx)
                            # if len(lx.shape)>1:
                            #     pr = lxx[:,self.num_inputs+self.num_actions:]
                            #     lxx = lxx[:,:self.num_inputs]
                            # else:
                            #     pr = lxx[self.num_inputs+self.num_actions:]
                            #     lxx = lxx[:self.num_inputs]
                            pr = self.reward_model(torch.cat([self.make_trj_input(lx, a),lh_trj[0][0].detach()], dim=-1))
                            #print('predicted r: ',pr)
                        else:
                            pr = r

                        pr = pr.to(device=lxx.device).squeeze(-1)

                        q_episodic[:,a] =  pr+self.args.gamma*self.episodic_net(h_trj_aa, is_learning)
                        #print(lx, a, q_episodic[:,a])
                        #print('next v', self.episodic_net(h_trj_aa, is_learning))
                        #print('cur v est, ', q_episodic[:,a])
                    if  is_learning is False and random.random()<self.args.bstr_rate:
                        curV = q_episodic.max(1)[0]
                        if len(curV.shape)==1:
                            curV = curV.unsqueeze(-1)

                        self.dnd.lookup2write(self.future_net(lh_trj[0][0]), curV, K=args.k_write)


                    #tem, train_var = self.dnd.lookup_grad(self.future_net(lh_trj[0][0]))
                    #loss = mse_criterion(curV.detach(), tem)
                    #self.optimizer_dnd.zero_grad()
                    #loss.backward(retain_graph=True)
                    #self.optimizer_dnd.step()


                    #for ii in range(len(train_var)):
                    #    #print(train_var[ii][0].grad.data)
                    #    train_var[ii][0].data += 0.001 * train_var[ii][0].grad.data
                    #    self.dnd.keys[train_var[ii][1]] = train_var[ii][0]
                    #    train_var[ii][0].grad.data.zero_()
                    #    ii+=1

                    #self.dnd.kdtree.build_index(self.dnd.keys.data.cpu().numpy())


            # q_episodic = self.episodic_net(h_trj)
        # if episode%5==0:
        #     print(f" semantic q {q_value_semantic[0]} vs episodic q {q_episodic[0]}")
        # q_value = self.act_net(torch.cat([q_value_semantic, q_episodic], dim=-1))
        q_episodic = q_episodic.detach()

        if args.fix_alpha>0:
            a = args.fix_alpha
        else:
            if target == 0:
                a = self.choice_net(h_trj[0][0])
                # a = self.alpha
            else:
                a = self.choice_net_target(h_trj[0][0])
                # a = self.alpha_target
            # a = self.alpha
            a = F.sigmoid(a)
        #q_value_semantic = self.semantic_net(torch.cat([x, a*q_episodic], dim=-1), target)

        #if random.random()<0.0003:
        #    print(q_value_semantic[0], q_episodic[0]*a[0])
        #a = F.sigmoid(self.choice_net(x))
        #a = F.tanh(self.alpha)
        # a=0
        # return self.act_net(q_episodic)
        # if target == 0:
        if self.args.td_interval>0:
            if target == 0:
               return q_episodic*a+q_value_semantic,q_value_semantic, q_episodic*a
            else:
               return q_episodic*a+q_value_semantic,q_value_semantic, q_episodic*a
            # if target == 0:
            #      return self.act_net(q_episodic*a+q_value_semantic),q_value_semantic, q_episodic*a
            # else:
            #      return self.act_net_target(q_episodic*a+q_value_semantic),q_value_semantic, q_episodic*a


            # return q_episodic*a+ q_value_semantic,q_value_semantic, q_episodic
        # else:
        #     return q_value_semantic, q_value_semantic, q_value_semantic
        # return q_value
        return q_episodic, q_episodic, q_episodic
        # return q_value_semantic*F.sigmoid(q_episodic),q_value_semantic, q_episodic
        # op = torch.matmul(q_value_semantic.unsqueeze(-1),F.sigmoid(q_episodic).unsqueeze(1))
        # q = torch.matmul(op, self.beta.unsqueeze(0).repeat(q_value_semantic.shape[0], 1).unsqueeze(-1))
        # return q.squeeze(-1),q_value_semantic, q_episodic


    def exploit(self,  x):
        batch_size = x.shape[0]
        q_episodics = []

        for (h_trj,v) in self.best_trj:
            z = torch.zeros(x.shape[0], self.num_actions)
            if USE_CUDA:
                z = z.cuda()
            kw = z
            last_h_trj = (h_trj[0].repeat(1, batch_size, 1), h_trj[1].repeat(1, batch_size, 1))

            for a in range(self.num_actions):
                X = self.make_trj_input(x, a)
                y_p, nh = self.trj_model(X, last_h_trj)
                y_p = self.trj_out(y_p)
                rec_loss = torch.norm(y_p[:,:self.num_actions+self.num_inputs]-
                                      X[:, :self.num_actions + self.num_inputs], dim=-1, keepdim=True)
                kw[:,a] = -rec_loss

            kw = torch.exp(kw)
            kw = kw/torch.sum(kw, dim=-1, keepdim=True)
            vs = kw*1.6**v
            q_episodics.append(vs)
        return torch.mean(torch.stack(q_episodics, dim=0), dim=0)

    def value(self, s, h=None):
        if h is None:
            h = self.trj_model.create_new_state(1)
            q, qs, qe = self.forward(s, h)
            return torch.max(q, dim=-1)[0]
        else:
            return self.episodic_net(h)

    def planning(self, x, h_trj, a=None, plan_step=1):
        actions = []
        qa = 0
        qsa = 0
        qea = 0

        t = 0
        for s in range(plan_step):
            q, qs, qe = self.forward(x, h_trj, a=a)
            action = q.max(1)[1].item()
            y_trj, h_trj = self.trj_model(self.make_trj_input(x, action), h_trj)
            actions.append(action)
            qa+=q*(self.args.gamma**t)
            qsa+=qs*(self.args.gamma**t)
            qea+=qe*(self.args.gamma**t)


            t+=1
            x = y_trj[:,:self.num_inputs]

        return qa, qsa, qea, actions

    def get_pivot_lastinsert(self):
        if len(self.last_inserts)>0:
            return min(self.last_inserts)
        else:
            return -10000000

    def semantic_net(self, x, target=0):
        if "maze_img" in self.args.task2:
            if "cnn" not in self.args.task2:
                x = self.proj(x).detach()
            else:
                x = self.cnn(x)
        elif "world" in self.args.task2:
            x = self.cnn(x)

        if target == 0:
            return self.qnet(x)
        else:
            return self.qnet_target(x)

    def episodic_net(self, h_trj, is_learning=False, K=0):
        fh_trj = self.future_net(h_trj[0][0])
        # fh_trj = h_trj[0][0].detach()
        q_estimates = self.dnd.lookup(fh_trj, is_learning=is_learning, K=K, p=args.pread)
        # print("q lookup", q_estimates)
        return q_estimates

    def make_trj_input(self, x, a, r=None):

        if "maze_img" in self.args.task2:
            if "cnn" not in self.args.task2:
                x = self.proj(x).detach()
            else:
                if len(x.shape)==3:
                    x = x.unsqueeze(0)

                x = self.cnn(x)
        elif "world" in self.args.task2:
            x = self.cnn(x)

        a_vec = torch.zeros(x.shape[0],self.num_actions)
        a_vec[:,a] = 1

        if USE_CUDA:
            a_vec = a_vec.cuda()
#            r = r.cuda()
        x = torch.cat([x, a_vec],dim=-1)

        return x


    def add_trj(self, h_trj, R, step, episode, action):
        # print(f"add R {R}")
        if self.args.write_interval<0:
            h_trj = h_trj
        else:
            h_trj = h_trj[0][0]
        hkey = torch.as_tensor(h_trj).float()#.detach()
        if USE_CUDA:
            hkey = hkey.cuda()
        hkey = self.future_net(hkey)
        # t = torch.Tensor([0.5])
        # hkey = (F.sigmoid(hkey) > t).float() * 1
        if self.args.write_interval<0:
            hkey = self.state2key(hkey.cuda())
            rvec = torch.zeros(1, self.num_actions)
            rvec[0,action] = R
            #print(hkey, rvec)
            #raise False
        else:
            rvec = R.unsqueeze(0)
        if USE_CUDA:
            rvec = rvec.cuda()

        # print(hkey)
        embedding_index = self.dnd.get_index(hkey)
        if embedding_index is None:
            self.dnd.insert(hkey, rvec.detach())

            if self.insert_size>0:
                if len(self.last_inserts) > self.insert_size:
                    self.last_inserts.sort()
                    self.last_inserts = self.last_inserts[1:-1]
                self.last_inserts.append(rvec.detach())
            if episode>self.num_warm_up and self.dnd.keys is not None:
                #try:
                self.dnd.lookup2write(hkey, rvec.detach())
                #except Exception as e:
                #    print(e)
        else:
            #print("dssssssss")
            if embedding_index not in self.emb_index2count:
                self.emb_index2count[embedding_index] = 1


            if self.args.write_interval<0:
                rvec = torch.zeros(1, self.num_actions)
                rvec[0, action] = R
                rvec = R.unsqueeze(0)
                if USE_CUDA:
                    rvec = rvec.cuda()
                self.dnd.update(torch.max(rvec,
                                          torch.tensor(self.emb_index2count[embedding_index]).float().unsqueeze(0).to(
                                              device=rvec.device)),
                                embedding_index)
            else:
                #R = self.dnd.values[embedding_index]*self.emb_index2count[embedding_index]+ R
                #R = R/(self.emb_index2count[embedding_index]+1)
                #self.emb_index2count[embedding_index]+=1
                #self.dnd.update(R.unsqueeze(0).detach(), embedding_index)

                # self.dnd.update(torch.max(R.unsqueeze(0),
                #                 torch.tensor(self.emb_index2count[embedding_index]).float().unsqueeze(0).to(device=R.device)),
                #                 embedding_index)

                if episode > self.num_warm_up and self.dnd.keys is not None:
                    # try:
                    self.dnd.lookup2write(hkey, rvec.detach(), K=args.k_write)

    def compute_rec_loss(self, last_h_trj, traj_buffer, optimizer, batch_size, noise=0.1):
        # print('len ', len(traj_buffer))
        sasr = random.choices(traj_buffer, k=batch_size-1)
        sasr.append(traj_buffer[-1])

        X = []
        y = []
        y2 = []
        hs1 = []
        hs2 = []
        for s1,h, a,s2,r,o_r in sasr:
            s1 = torch.as_tensor(s1).float()
            s2 = torch.as_tensor(s2).float()
            if USE_CUDA:
                s1 = s1.cuda()
                s2 = s2.cuda()
            if len(s1.shape)==1 or len(s1.shape)==3:
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)

            o_r = torch.FloatTensor([o_r]).unsqueeze(0)
            r = torch.FloatTensor([r]).unsqueeze(0)

            x = self.make_trj_input(s1, a, o_r)
            x2 = self.make_trj_input(s2, a, o_r)

            #X.append(x)
            if noise>0:
                if random.random()>0.5:
                    X.append(F.dropout(x, p=noise))
                else:
                    noise_tensor = ((torch.max(torch.abs(x))*noise)**0.5)*torch.randn(x.shape)
                    if USE_CUDA:
                       noise_tensor = torch.tensor(noise_tensor).cuda()
                    X.append(x + noise_tensor.float())
            else:
                X.append(x)
            y.append(x)

            y2.append(torch.cat([x2, r.to(device=x2.device)], dim=-1))

            hs1.append(torch.tensor(h[0]).to(device=last_h_trj[0].device))
            hs2.append(torch.tensor(h[1]).to(device=last_h_trj[0].device))
        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        y2 = torch.stack(y2, dim=0).squeeze(1)

        last_h_trj = (last_h_trj[0].repeat(1, batch_size, 1), last_h_trj[1].repeat(1, batch_size, 1))
        cur_h_trj = (torch.cat(hs1, dim=1),
                     torch.cat(hs2, dim=1))

        if args.rec_type=="pred":
            last_h_trj = cur_h_trj
        y_p, _ = self.trj_model(X.squeeze(1), last_h_trj)
        y_p = self.trj_out(y_p)
       # pr = self.reward_model(torch.cat([X.squeeze(1), cur_h_trj],dim=-1))
        _, h_trj = self.trj_model(y.squeeze(1), cur_h_trj)
        # h_pred = self.future_net(h_trj[0][0])
        h_pred = h_trj[0][0]
        #print(y_p.shape)
        #print(X)
        #print(y2[:, self.num_inputs + self.num_actions:])
        l1 = mse_criterion(y_p[:, :self.num_inputs + self.num_actions], y2[:, :self.num_inputs + self.num_actions])
        # l2 = mse_criterion(y_p[:,self.num_inputs+self.num_actions:], y2[:,self.num_inputs+self.num_actions:])
        #l2 = mse_criterion(pr, y2[:, self.num_inputs + self.num_actions:])
        l3 = mse_criterion(cur_h_trj[0][0], last_h_trj[0][0])
        loss =  l1
        #loss = loss + l2
        # loss =  loss + l3
        # loss = mse_criterion(h_pred, last_h_trj[0][0])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.parameters(), 10)
        optimizer.step()

        return loss, l1, 0, l3

    def compute_reward_loss(self, last_h_trj, traj_buffer, optimizer, batch_size, noise=0.1):
        # print('len ', len(traj_buffer))
        sasr = random.choices(traj_buffer, k=batch_size-1)
        sasr.append(traj_buffer[-1])

        X = []
        y = []
        y2 = []
        hs1 = []
        hs2 = []
        for s1,h, a,s2,r,o_r in sasr:
            s1 = torch.as_tensor(s1).float()
            s2 = torch.as_tensor(s2).float()
            if USE_CUDA:
                s1 = s1.cuda()
                s2 = s2.cuda()
            if len(s1.shape)==1 or len(s1.shape)==3:
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)

            o_r = torch.FloatTensor([o_r]).unsqueeze(0)
            r = torch.FloatTensor([r]).unsqueeze(0)

            x = self.make_trj_input(s1, a, o_r)
            x2 = self.make_trj_input(s2, a, o_r)

            #X.append(x)
            if noise>0:
                if random.random()>0.5:
                    X.append(F.dropout(x, p=noise))
                else:
                    noise_tensor = ((torch.max(torch.abs(x))*noise)**0.5)*torch.randn(x.shape)
                    if USE_CUDA:
                       noise_tensor = torch.tensor(noise_tensor).cuda()
                    X.append(x + noise_tensor.float())
            else:
                X.append(x)

            y.append(x)

            y2.append(torch.cat([x2, r.to(device=x2.device)], dim=-1))

            hs1.append(torch.tensor(h[0]).to(device=last_h_trj[0].device))
            hs2.append(torch.tensor(h[1]).to(device=last_h_trj[0].device))
        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        y2 = torch.stack(y2, dim=0).squeeze(1)
        cur_h_trj = (torch.cat(hs1, dim=1),
                     torch.cat(hs2, dim=1))


        # print(X)
        # print(y2[:, self.num_inputs + self.num_actions:])

        pr = self.reward_model(torch.cat([X.squeeze(1), cur_h_trj[0][0]],dim=-1))
        l2 = mse_criterion(pr, y2[:, self.num_inputs + self.num_actions:])
        optimizer.zero_grad()
        l2.backward()
        optimizer.step()
        return l2

    def compute_td_loss(self, optimizer, batch_size, episode=0):
        state, h_trj, action, reward, old_reward, next_state, nh_trj, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        old_reward = Variable(torch.FloatTensor(old_reward))

        done = Variable(torch.FloatTensor(done))

        if USE_CUDA:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            old_reward = old_reward.cuda()
            done = done.cuda()


        # print(h_trj)
        # print(torch.tensor(h_trj[:,0,0,0]))
        hx = torch.tensor(h_trj[:,0,0,0]).to(device=state.device).unsqueeze(0)#torch.cat(torch.tensor(h_trj[:,0,0]).tolist(), dim=1)
        cx = torch.tensor(h_trj[:,1,0,0]).to(device=state.device).unsqueeze(0)#torch.cat(torch.tensor(h_trj[:,1,0]).tolist(), dim=1)
        q_values, q1, q2 = self.forward(state, (hx, cx), episode, use_mem=1, target=0, r=reward.unsqueeze(-1), a=action, is_learning=True)
        nhx = torch.tensor(nh_trj[:,0,0,0]).to(device=state.device).unsqueeze(0)#torch.cat(torch.tensor(nh_trj[:, 0,0]).tolist(), dim=1)
        ncx = torch.tensor(nh_trj[:,1,0,0]).to(device=state.device).unsqueeze(0)#torch.cat(torch.tensor(nh_trj[:, 1,0]).tolist(), dim=1)
        # raction = torch.randint(0, self.num_actions, action.shape)
        # if USE_CUDA:
        #     raction = raction.cuda()
        # next_q_values = self.forward(next_state, (nhx, ncx), episode, use_mem=1)
        next_q_values, qn1, qn2 = self.forward(next_state, (nhx, ncx), episode, use_mem=1, target=1, r=None, is_learning=True)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # q_value2 = q2.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        # next_q_value = next_q_values.gather(1, raction.unsqueeze(1)).squeeze(1)

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        # next_q_value2 = qn2.max(1)[0]
        # expected_q_value2 = reward + self.gamma * next_q_value2 * (1 - done)
        # print(q_value)
        # print(expected_q_value)
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        # loss = (expected_q_value-q_value2).pow(2).mean()
        # loss = loss + (q1-q2).pow(2).mean()
        # loss = loss + (qn1/qn2-1).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.args.clip)
        optimizer.step()

        return loss


    def act(self, state, h_trj, epsilon, r=0, episode=0):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        actions = actione = None
        if random.random() > epsilon and self.dnd.get_mem_size()>1:
            q_value, qs, qe = self.forward(state, h_trj, episode,
                                         r=None,
                                         use_mem=1)

            #q_value, qs, qe, _ = self.planning(state, h_trj, plan_step=2)

            action = q_value.max(1)[1].data[0]
            actions = qs.max(1)[1].data[0]
            actione = qe.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)

        y_trj, h_trj = self.trj_model(self.make_trj_input(state, action,
                                                          torch.FloatTensor([r]).unsqueeze(0)),
                                                          h_trj)
        # print("act {}".format(h_trj[0].shape))
        # y_trj = self.trj_out(y_trj)
        return action, h_trj, y_trj, actions, actione

    def update_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.choice_net_target.load_state_dict(self.choice_net.state_dict())
        self.act_net_target.load_state_dict(self.act_net.state_dict())
        self.alpha_target = self.alpha


# def plot(frame_idx, rewards, td_losses, rec_losses):
#     # clear_output(True)
#     # plt.figure(figsize=(20,5))
#     # plt.subplot(121)
#     # plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
#     # plt.plot(rewards)
#     # plt.subplot(122)
#     # plt.title('loss')
#     # plt.plot(losses)
#     # plt.show()







def run(args):
    epsilon_start = args.epsilon_start
    epsilon_final = args.epsilon_final
    epsilon_decay = args.epsilon_decay

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    batch_size = args.batch_size
    args.replay_buffer = ReplayBuffer(args.replay_size)
    env_id = args.task


    if "world" in args.task2:
        env = gym.make(env_id)

        env=img_featurize.FrameStack(env, 4)
        env = img_featurize.ImageToPyTorch(env)
    else:
        env = gym.make(env_id)


    if args.mode=="train":
        log_dir = f'./log/{args.task}{args.task2}new/{args.model_name}' \
            f'k{args.k}n{args.memory_size}i{args.write_interval}w{args.num_warm_up}h{args.hidden_size}' \
            f'u{args.update_interval}b{args.replay_size}l{args.insert_size}a{args.fix_alpha}' \
            f'm{args.min_reward}q{args.qnet_size}-{args.run_id}/'
        print(log_dir)
        configure(log_dir)

    model = DQN_DTM(env, args)
    num_params = 0
    for p in model.parameters():
        num_params += p.data.view(-1).size(0)
    print(f"no params {num_params}")

    if USE_CUDA:
        model = model.cuda()

    model.update_target()

    num_frames = args.n_epochs
    optimizer_td = optim.Adam(model.parameters(), lr=args.lr)
    scheduler_td = StepLR(optimizer_td, step_size=500, gamma=args.decay)

    optimizer_reward = optim.Adam(model.reward_model.parameters())

    optimizer_rec = optim.Adam(model.parameters(), lr=args.lr)

    td_losses = []
    rec_losses = []
    rec_l1s = []
    rec_l2s = []
    rec_l3s = []

    all_rewards = []
    episode_reward = mepisode_reward = 0
    traj_buffer = []
    state_buffer = []
    best_reward = -1000000000

    state = env.reset()
    ostate = state.copy()



    if "maze_img" in args.task2:
        state_map = {}
        img = env.render("rgb_array")
        if "view" in args.task2:
            img = img_featurize.get_viewport(img, state, num_state=(env.observation_space.high[0]+1)**2)
        if "cnn" not in args.task2:
            # Load the pretrained model
            stateim = img_featurize.get_vector(img)
            stateim = np.concatenate([stateim, state])

            state_map[tuple(state.tolist())] = stateim
            state = state_map[tuple(state.tolist())]
        else:
            state_map[tuple(state.tolist())] = img_featurize.get_image(img)
            state =  state_map[tuple(state.tolist())]

    if "world" in args.task2:
        state = img_featurize.get_image2(state)


    if "trap" in args.task2:
        trap = [0, 0]
        while trap[0] == 0 and trap[1] == 0 or trap[0] == env.observation_space.high[0] and trap[1] == \
                env.observation_space.high[0]:
            trap = [random.randint(0, env.observation_space.high[0]),
                    random.randint(0, env.observation_space.high[0])]
        print("trap ", trap)
        state = np.concatenate([state, trap])

    episode_num = 0
    best_episode_reward = -1e9
    step = 0
    h_trj = model.trj_model.create_new_state(1)
    reward = old_reward = 0
    m_contr = []
    s_contr = []

    for frame_idx in tqdm(range(1, num_frames + 1)):

        epsilon = epsilon_by_frame(frame_idx)
        action, nh_trj, y_trj, action_s, action_e = model.act(state, h_trj, epsilon, r=reward, episode=episode_num)

        try:
            action = action.item()
        except:
            pass
        if action_e is not None:
            if action == action_e:
                m_contr.append(1)
            else:
                m_contr.append(0)

            if action == action_s:
                s_contr.append(1)
            else:
                s_contr.append(0)

        old_reward = reward
        next_state, reward, done, _ = env.step(action)
        reward = float(reward)




        if "maze" in args.task:
            if step > 1000 and "hard" in args.task2:
                done = 1
            if next_state[0] == ostate[0] and next_state[1] == ostate[1]:
                reward = reward - 1
                if "hard" in args.task2:
                    next_state = env.reset()
            if "trap" in args.task2 and next_state[0]==trap[0] and next_state[1]==trap[1]:
                reward = reward - 2
            if "trap_key" in args.task2 and next_state[0]==1 and next_state[1]==1:
                trap = [-1, -1]
                print("free trap")

        #print(next_state,action, state, reward)


        m_reward = reward

        #if "world" in args.task2:
        #    if reward<1.0:
        #        reward=-0.01

        if args.rnoise>0:
            reward += np.random.normal(0, args.rnoise, 1)[0]
        if -1<args.rnoise < 0:
            if random.random() > -args.rnoise:
                reward = -reward

        if random.random()<args.pnoise:
            next_state = state.copy()

        if args.render==1:
            env.render()



        ostate = next_state.copy()

        if "maze_img" in args.task2:

            if tuple(next_state.tolist()) in state_map:
                next_state = state_map[tuple(next_state.tolist())]
            else:
                img = env.render("rgb_array")
                if "view" in args.task2:
                    img = img_featurize.get_viewport(img, next_state, num_state=(env.observation_space.high[0]+1)**2)
                # Load the pretrained model
                if "cnn" not in args.task2:
                    next_stateim = img_featurize.get_vector(img)
                    next_stateim = np.concatenate([next_stateim, next_state])

                    state_map[tuple(next_state.tolist())] = next_stateim
                    next_state = next_stateim
                else:
                    state_map[tuple(next_state.tolist())] = img_featurize.get_image(img)
                    next_state = state_map[tuple(next_state.tolist())]

        if "world" in args.task2:
            next_state = img_featurize.get_image2(next_state)

        if "trap" in args.task2:
            next_state = np.concatenate([next_state, trap])

        with torch.no_grad():

            args.replay_buffer.push(state,  (h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()),
                                    action, reward, old_reward, next_state,
                                    (nh_trj[0].detach().cpu().numpy(),nh_trj[1].detach().cpu().numpy()), done)
        traj_buffer.append((state.copy(), (h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()), action, next_state.copy(), reward, old_reward))
        h_trj = nh_trj

        # traj_buffer.append((next_state, action, state ))

        state = next_state.copy()
        mepisode_reward += m_reward
        episode_reward += reward
        step+=1
        if done:
            # print(f"done e {episode_num}")

            if args.write_interval<0:
                state_buffer.append(([state.copy()], episode_reward, step, action))
            else:
                state_buffer.append(((h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()), episode_reward, step, action))

            add_mem = 0

            cc = 0

            for h, R, trj_step, action in state_buffer:
                RR = episode_reward - R
                if  RR>args.min_reward:
                    if len(model.last_inserts)>=args.insert_size>0 and RR<model.get_pivot_lastinsert():# and RR<max(model.last_inserts):
                        pass
                    else:
                        model.add_trj(h,  torch.as_tensor(RR), trj_step, episode_num, action)
                        add_mem=1

                cc+=1


            if episode_reward>best_episode_reward:
                best_episode_reward = episode_reward

            # model.best_trj.append((h_trj, episode_reward))
            # if len(model.best_trj)>10:
            #     model.best_trj = sorted(model.best_trj, key=lambda tup: tup[1])
            #     model.best_trj = model.best_trj[1:]
            #

            l2 = model.compute_reward_loss(h_trj, traj_buffer, optimizer_reward, args.batch_size_reward,
                                           noise=0)
            rec_l2s.append(l2.data.item())

            if  frame_idx < args.rec_period and args.rec_rate>0:
                loss, l1, l2, l3 = model.compute_rec_loss(h_trj, traj_buffer, optimizer_rec, args.batch_size_plan,
                                                          noise=args.rec_noise)

                rec_losses.append(loss.data.item())
                rec_l1s.append(l1.data.item())
                rec_l3s.append(l3.data.item())

            if add_mem==1:
                model.dnd.commit_insert()

            h_trj = model.trj_model.create_new_state(1)
            state = env.reset()
            if "maze_img" in args.task2:
                state = state_map[tuple(state.tolist())]


            all_rewards.append(mepisode_reward)
            #print("episode reward", episode_reward)

            episode_reward = mepisode_reward = 0
            del traj_buffer
            del state_buffer

            traj_buffer = []
            state_buffer = []
            episode_num+=1
            step=0
            if "random" in args.task2:
               env.close()
               del env
               env = gym.make(env_id)
               state = env.reset()
               ostate = state.copy()
               if "maze_img" in args.task2:
                   state_map = {}
                   img = env.render("rgb_array")
                   if "view" in args.task2:
                       img = img_featurize.get_viewport(img, state, num_state=(env.observation_space.high[0]+1)**2)
                   if "cnn" not in args.task2:
                       # Load the pretrained model
                       stateim = img_featurize.get_vector(img)
                       stateim = np.concatenate([stateim, state])

                       state_map[tuple(state.tolist())] = stateim
                       state = state_map[tuple(state.tolist())]
                   else:
                        state_map[tuple(state.tolist())] = img_featurize.get_image(img)
                        state = state_map[tuple(state.tolist())]

            if "world" in args.task2:
                state = img_featurize.get_image2(state)

            if "trap" in args.task2:
                trap = [0, 0]
                while trap[0] == 0 and trap[1] == 0 or trap[0] == env.observation_space.high[0] and trap[1] == \
                        env.observation_space.high[0]:
                    trap = [random.randint(0, env.observation_space.high[0]),
                            random.randint(0, env.observation_space.high[0])]
                print("trap ", trap)
                state = np.concatenate([state, trap])
            log_value('Reward/episode reward', np.mean(all_rewards[-args.num_avg_reward:]), episode_num)


        else:
            if args.write_interval<0:
                state_buffer.append(([state.copy()], episode_reward, step, action))

            elif frame_idx % args.write_interval == 0 and len(traj_buffer) > 0:
                #if random.random() < args.rec_rate:
                l2 = model.compute_reward_loss(h_trj, traj_buffer, optimizer_reward, args.batch_size_reward,
                                                   noise=0)
                rec_l2s.append(l2.data.item())
                if random.random() < args.rec_rate and frame_idx<args.rec_period:
                    loss, l1,l2,l3 = model.compute_rec_loss(h_trj, traj_buffer, optimizer_rec, args.batch_size_plan, noise = args.rec_noise)
                    rec_losses.append(loss.data.item())
                    rec_l1s.append(l1.data.item())
                    #rec_l2s.append(l2.data.item())
                    rec_l3s.append(l3.data.item())
                state_buffer.append(((h_trj[0].detach().cpu().numpy(),h_trj[1].detach().cpu().numpy()), episode_reward, step, action))
                # h_trj = model.trj_model.create_new_state(1)
                # h_trj = (h_trj[0].detach(), h_trj[1].detach())

        if args.td_interval>0 and frame_idx % args.td_interval == 0 and frame_idx > args.td_start and len(args.replay_buffer) > batch_size:
            loss = model.compute_td_loss(optimizer_td, batch_size, episode_num)
            scheduler_td.step()
            td_losses.append(loss.data.item())



        # if frame_idx % 2000 == 0 and frame_idx > 0:
        #     for param_group in optimizer_rec.param_groups:
        #         param_group['lr'] = param_group['lr'] / 2
        #     for param_group in optimizer_td.param_groups:
        #         param_group['lr'] = param_group['lr'] / 2

        if frame_idx % args.update_interval == 0:
            model.update_target()

        if frame_idx % args.plot_interval == 0:
            #print(optimizer_td.param_groups[0]['lr'])
            log_value('Mem/num stored', model.dnd.get_mem_size(), int(frame_idx))
            # log_value('Mem/alpha', model.alpha.detach().item(), int(frame_idx))
            log_value('Mem/min last', model.get_pivot_lastinsert(), int(frame_idx))
            log_value('Mem/contrib', np.mean(m_contr), int(frame_idx))
            log_value('Mem/scontrib', np.mean(s_contr), int(frame_idx))

            log_value('Loss/rec loss', np.mean(rec_losses), int(frame_idx))
            log_value('Loss/rec l1 loss', np.mean(rec_l1s), int(frame_idx))
            log_value('Loss/rec l2 loss', np.mean(rec_l2s), int(frame_idx))
            log_value('Loss/rec l3 loss', np.mean(rec_l3s), int(frame_idx))
            log_value('Loss/td loss', np.mean(td_losses), int(frame_idx))
            log_value('Episode/num episode', episode_num, int(frame_idx))

            currw = np.mean(all_rewards[-args.num_avg_reward:])

            log_value('Reward/frame reward', currw, int(frame_idx))
            print(f'episode {episode_num} step {frame_idx} avg rewards {currw} vs best {best_reward}')

            if best_reward<currw or best_reward>=currw and frame_idx>num_frames-2* args.plot_interval:
                best_reward = currw
                if os.path.isdir(args.save_model) is False:
                    os.mkdir(args.save_model)
                save_dir = os.path.join(args.save_model, args.task)
                if os.path.isdir(save_dir) is False:
                    os.mkdir(save_dir)
                with open(os.path.join(save_dir, f'args.jon'), 'w') as fp:
                    sa = copy.copy(args)
                    sa.device = None
                    sa.replay_buffer = None
                    json.dump(vars(sa), fp)
                save_dir = os.path.join(save_dir, f"{args.model_name}.pt")
                torch.save(model.state_dict(), save_dir)
                print(f"save model to {save_dir}!")

                with open(save_dir + 'mem', 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(model.dnd, output)
                print(f"save memory to {save_dir + 'mem'}!")

            if len(all_rewards) > args.num_avg_reward*1.1:
                all_rewards = all_rewards[-args.num_avg_reward:]
            td_losses = []
            rec_losses = []
            rec_l1s = []
            rec_l2s = []
            rec_l3s = []
            m_contr = []
            s_contr = []

            if episode_num > args.max_episode > 0:
                break



if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for TEM.")
    parser.add_argument("--mode", default="train",
                        help="train or test")
    parser.add_argument("--save_model", default="./model/",
                        help="save model dir")
    parser.add_argument("--rnoise", type=float, default=0,
                        help="add noise to reward")
    parser.add_argument("--pnoise", type=float, default=0,
                        help="add noise to state")
    parser.add_argument("--task", default="MiniWorld-CollectHealth-v0",
                        help="task name")
    parser.add_argument("--task2", default="miniworld",
                        help="task name")
    parser.add_argument("--render", type=int, default=0,
                        help="render or not")
    parser.add_argument("--n_epochs", type=int, default=1000000,
                        help="number of epochs")
    parser.add_argument("--max_episode", type=int, default=10000,
                        help="number of episode allowed")
    parser.add_argument("--model_name", default="DTM",
                        help="DTM")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="lr")
    parser.add_argument("--decay", type=float, default=1,
                        help=" decay lr")
    parser.add_argument("--clip", type=float, default=10,
                        help="clip gradient")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="exploration start")
    parser.add_argument("--epsilon_final", type=float, default=0.01,
                        help="exploration final")
    parser.add_argument("--epsilon_decay", type=float, default=500.0,
                        help="exploration decay")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="gamma")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size value model")
    parser.add_argument("--batch_size_plan", type=int, default=4,
                        help="batch size planning model")
    parser.add_argument("--batch_size_reward", type=int, default=32,
                        help="batch size planning model")
    parser.add_argument("--reward_hidden_size", type=int, default=32,
                        help="batch size planning model")
    parser.add_argument("--replay_size", type=int, default=1000000,
                        help="replay buffer size")
    parser.add_argument("--qnet_size", type=int, default=128,
                        help="MLP hidden")
    parser.add_argument("--hidden_size", type=int, default=16,
                        help="RNN hidden")
    parser.add_argument("--mem_dim", type=int, default=5,
                       help="memory dimesntion")
    parser.add_argument("--memory_size", type=int, default=10000,
                        help="memory size")
    parser.add_argument("--insert_size", type=int, default=-1,
                        help="insert size")
    parser.add_argument("--k", type=int, default=5,
                        help="num neighbor")
    parser.add_argument("--k_write", type=int, default=-1,
                        help="num neighbor")
    parser.add_argument("--mem_mode", type=int, default=0,
                        help="memory_mode")
    parser.add_argument("--pread", type=float, default=0.7,
                        help="minimum reward of env")
    parser.add_argument("--min_reward", type=float, default=-1e8,
                        help="minimum reward of env")
    parser.add_argument("--write_interval", type=int, default=10,
                        help="interval for memory writing")
    parser.add_argument("--write_lr", type=float, default=.5,
                        help="learning rate of writing")
    parser.add_argument("--fix_alpha", type=float, default=-1,
                        help="fix alpha")
    parser.add_argument("--bstr_rate", type=float, default=0.1,
                        help="learning rate of writing")
    parser.add_argument("--td_interval", type=int, default=1,
                        help="interval for td update")
    parser.add_argument("--td_start", type=int, default=-1,
                        help="interval for td update")
    parser.add_argument("--rec_rate", type=float, default=0.1,
                        help="rate of reconstruction learning")
    parser.add_argument("--rec_noise", type=float, default=0.1,
                        help="rate of reconstruction learning")
    parser.add_argument("--rec_type", type=str, default="mem",
                        help="rec type")
    parser.add_argument("--rec_period", type=int, default=1e40,
                        help="period of reconstruction learning")
    parser.add_argument("--update_interval", type=int, default=100,
                        help="interval for update target Q network")
    parser.add_argument("--plot_interval", type=int, default=200,
                        help="interval for plotting")
    parser.add_argument("--num_avg_reward", type=int, default=10,
                        help="interval for plotting")
    parser.add_argument("--num_warm_up", type=int, default=-1,
                        help="number of episode warming up memory")
    parser.add_argument("--run_id", default="no_td",
                        help="r1,r2,r3")

    global args
    args  = parser.parse_args()
    #import sys
    #with open(f'./log_screen_mem{args.model_name}.txt', 'w') as f:
    if args.k_write<0:
        args.k_write=args.k
    run(args)