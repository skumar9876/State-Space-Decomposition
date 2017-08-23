import random
import numpy as np

"""
Map Values:
Red = 20
Green = 30
Blue = 40
Yellow = 50
Agent = 10
"""

class World():

    def __init__(self, fixed=False, attention_used=True):
        self.fixed = fixed
        if attention_used == True:
            self.reset()
        else:
            self.reset_no_attention()


    def reset(self):
        #self.colors_dict = {20: 0, 30: 1, 40: 2, 50: 3}
        self.colors_dict = {63: 0, 126: 1, 189: 2, 252: 3}
        self.colors_dict_inverted = {0: 63, 1: 126, 2: 189, 3: 252}


        self.world, self.sentence, self.color = self.generate_map()

        self.attention = np.array([0, 0])

        self.update_attention()
        self.num_steps = 0

        #self.MAX_STEPS = 50
        #self.MAX_STEPS = 100
        if self.fixed == False:
            self.MAX_STEPS = 1000
        else:
            self.MAX_STEPS = 1000




        return self.attention_image.copy(), self.sentence.copy()


    def reset_no_attention(self):
        self.colors_dict = {63: 0, 126: 1, 189: 2, 252: 3}
        self.colors_dict_inverted = {0: 63, 1: 126, 2: 189, 3: 252}

        self.world, self.sentence, self.color = self.generate_map()

        self.num_steps = 0

        self.MAX_STEPS = 100

        return self.world.copy(), self.sentence.copy()


    def generate_map(self):
        world = []
        colors = self.colors_dict.keys()

        if self.fixed == False:
            random.shuffle(colors) # Randomly shuffle the colors

        lines = 10

        fixed_num_lines = [3, 2, 3, 2]

        for i in xrange(len(colors)):
            color = colors[i]
            num_lines = 0

            if i == len(colors) - 1: # This is the last color
                num_lines = lines
            else:
                if self.fixed == False:
                    num_lines = random.randint(1, 3)#random.randint(1, 4)
                    while lines - num_lines < 0:
                        num_lines = random.randint(1, 4)
                    lines -= num_lines
                else:
                    num_lines = fixed_num_lines[i]
                    lines -= num_lines

            for y in range(num_lines):
                for x in range(10):
                    world.append(color)


        


        # Room color at very bottom
        index = world[95]
        # Create the target sentence
        sentence_index = self.colors_dict[index]
        sentence = np.zeros([1, 4])
        sentence[0][sentence_index] = 1


        # Agent starting position
        self.x = 0
        self.y = 0
        world = np.array(world)
        world = world.reshape((10, 10))


        old_color = world[self.y][self.x]
        world[self.y][self.x] = 10


        return world, sentence, old_color

    # Actions: Attention Action, Sentence Action
    # 0 -> up, 20
    # 1 -> up, 30
    # 2 -> up, 40
    def convert_action(self, num):
        attention_action_dict = {0: 0, 1: 0, 2: 0, 3: 0,
                                 4: 1, 5: 1, 6: 1, 7: 1, 
                                 8: 2, 9: 2, 10:2, 11: 2}

        sentence_action_dict = {0: 0, 1: 1, 2: 2, 3: 3, 
                                4: 0, 5: 1, 6: 2, 7: 3, 
                                8: 0, 9: 1, 10:2, 11: 3}

        return attention_action_dict[num], sentence_action_dict[num]

    def next_attention(self, action):
        """
        action = [up, down, do nothing (optional action)]
        """
        y = self.attention[0]
        x = self.attention[1]
        if action[0] == 1:
            y = max(0, y - 1)
        if action[1] == 1:
            y = min(4, y + 1)

        self.attention = np.array([y, x])

    def update_attention(self):
        """
        Updates the 5x5 attention image
        """
        y = self.attention[0]
        x = self.attention[1]
        self.attention_image = np.array(self.world.copy())[y:y+5, x:x+5]

    def move_agent(self, sentence):
        """
        sentence = [20, 30, 40, 50, do nothing]
        """

        if sentence.index(1) < 4: #4 is the do nothing instruction
            goal = self.colors_dict_inverted[sentence.index(1)]

            action = 0

            for y in range(0, 10):
                color = self.world[y][self.x]
                if color == goal:
                    action = y
                    break

            self.world[self.y][self.x] = self.color

            if self.y < y:
                self.y += 1
            if self.y > y:
                self.y -= 1

            self.color = self.world[self.y][self.x]
            self.world[self.y][self.x] = 10

    def agent_in_attention_view(self):

        if self.y < self.attention[0] + 5 and self.y >= self.attention[0] \
        and self.x >= self.attention[1] and self.x < self.attention[1] + 5:
            return True
        else:
            return False

    def move_agent_strict(self, sentence):
        """
        sentence = [20, 30, 40, 50, do nothing]
        """

        if self.agent_in_attention_view(): #4 is the do nothing instruction
            goal = self.colors_dict_inverted[sentence.index(1)]

            action = 0

            for y in range(0, 10):
                color = self.world[y][self.x]
                if color == goal:
                    action = y
                    break

            # Check that target room is in attention view
            if y >= self.attention[0] + 5 or y < self.attention[0]:
                return

            else:
                self.world[self.y][self.x] = self.color

                if self.y < y:
                    self.y += 1
                if self.y > y:
                    self.y -= 1

                
                self.color = self.world[self.y][self.x]
                self.world[self.y][self.x] = 10


    def step_no_attention(self, chosen_room):
        sentence = [0, 0, 0, 0]
        sentence[chosen_room] = 1

        self.move_agent(sentence)

        self.num_steps += 1

        done = self.isTerminal(False)

        reward = self.reward()


        return self.world.copy(), self.sentence.copy(), reward, done

    def step(self, action, strict=False):
        attention_action = self.convert_action(action)[0]
        chosen_room = self.convert_action(action)[1]


        sentence = [0, 0, 0, 0, 0]
        sentence[chosen_room] = 1

        action = [0, 0, 0, 0, 0]
        action[attention_action] = 1

        if strict == True:
            self.move_agent_strict(sentence)
        else:
            self.move_agent(sentence)
        self.next_attention(action)

        self.update_attention()
        self.num_steps += 1


        done = self.isTerminal(strict)

        reward = self.reward()

        return self.attention_image.copy(), self.sentence.copy(), reward, done

    def step_pg(self, chosen_room, attention_action, strict=False):

        sentence = [0, 0, 0, 0, 0]
        sentence[chosen_room] = 1

        action = [0, 0, 0, 0, 0]
        action[attention_action] = 1

        if strict == True:
            self.move_agent_strict(sentence)
        else:
            self.move_agent(sentence)
        self.next_attention(action)

        self.update_attention()
        self.num_steps += 1


        done = self.isTerminal(strict)

        reward = self.reward()

        return self.attention_image.copy(), self.sentence.copy(), reward, done


    def episode_reward(self):
        if self.sentence[0][self.colors_dict[self.color]] == 1:
            #print "+1 reward"
            print self.num_steps

            return 1 - float(self.num_steps) / (float(self.MAX_STEPS) / 2)
        else:
            print "BAD REWARD"
            return 1 - float(self.num_steps) / (float(self.MAX_STEPS) / 2)

    def reward(self):
        if self.sentence[0][self.colors_dict[self.color]] == 1:
            return 1
            #return 100
        else:
            step_cost = -0.01 #- 2. / (float(self.MAX_STEPS))
            return step_cost 

    def isTerminal(self, strict):

        if strict == True:
                                                #Agent is on target color AND agent is in attention's view
            if self.num_steps >= self.MAX_STEPS or (self.sentence[0][self.colors_dict[self.color]] == 1 and self.agent_in_attention_view()):
                return True
            else:
                return False

        else:

            if self.num_steps >= self.MAX_STEPS or self.sentence[0][self.colors_dict[self.color]] == 1:
                return True
            else:
                return False

