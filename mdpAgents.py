# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import copy

class MDPAgent(Agent):

    def __init__(self):
        """
        Constructor: this gets run when we first invoke pacman.py
        """
        print "Starting up MDPAgent!"
        name = "Pacman"


    def registerInitialState(self, state):
        """
        Gets run after an MDPAgent object is created and once there is
        game state to access.
        """
        print "Running registerInitialState for MDPAgent!"

        # create map and add walls, rewards and utilities to it 
        self.make_map(state)
        self.add_walls(state)
        self.set_food(state)
        self.update_ghosts(state)
        self.update_utilities()

    def final(self, state):
        """
        This is what gets run in between multiple games
        """
        print "Looks like the game just ended!"

    
#########################################################
# Functions that create map
#########################################################
    def make_map(self,state):
        """
        Make a map by creating a grid of the right size
        """
        corners = api.corners(state)
        height = self.getLayoutHeight(corners)
        width  = self.getLayoutWidth(corners)
        self.map = Grid(width, height)
        
    def getLayoutHeight(self, corners):
        """
        Get the height of the grid.
        we add one to the value returned by corners to switch from the
        index (returned by corners) to the size of the grid
        """
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1

    def getLayoutWidth(self, corners):
        """
        Get the width of the grid.
        we add one to the value returned by corners to switch from the
        index (returned by corners) to the size of the grid
        """
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1

#########################################################
# Functions that manipulate the map.
#########################################################

    def add_walls(self, state):
        """
        Set grid squares as walls.

        :param state: current game state
        """
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.set_wall((walls[i][0], walls[i][1]))

    def set_food(self, state):
        """
        Update the reward of squares that have food.

        :param state: current game state
        """ 
        food = api.food(state) + api.capsules(state)
        for x,y in food:
            self.map.set_reward((x, y), 5)
    
     
    def update_ghosts(self, state):
        """
        Update the reward of squares where a ghost is in right now. 
        Locations close to it are affected as well.
        In case the ghost is scared the reward is positive and locations
        close to it are not affected

        :param state: current game state
        """ 
        ghosts = api.ghostStates(state)
        for pos, state in ghosts:
            if state == 0:
                self.map.set_reward((int(pos[0]), int(pos[1])), -10)
                self.map.set_reward((int(pos[0])-1, int(pos[1])+1), -5)
                self.map.set_reward((int(pos[0]), int(pos[1])+1), -5)
                self.map.set_reward((int(pos[0]+1), int(pos[1])+1), -5)
                self.map.set_reward((int(pos[0])-1, int(pos[1])), -5)
                self.map.set_reward((int(pos[0])+1, int(pos[1])), -5)
                self.map.set_reward((int(pos[0])-1, int(pos[1])-1), -5)
                self.map.set_reward((int(pos[0]), int(pos[1])-1), -5)
                self.map.set_reward((int(pos[0])+1, int(pos[1])-1), -5)
            else:
                self.map.set_reward((int(pos[0]), int(pos[1])), 20)
    
    def clean_grid_reward(self):
        """
        Set all grid squares's rewards as 0.
        """
        for i in range(self.map.get_width()):
            for j in range(self.map.get_height()):
                self.map.set_reward((i, j), 0)

#########################################################
# Functions related to Value Interation
#########################################################
    def update_utilities(self):
        """
        Update utilities of each grid square.

        :return: None. It returns when the utilities converge.
        """ 
        for x in range(50):
            new_map = copy.deepcopy(self.map)

            for i in range(self.map.get_width()):
                for j in range(self.map.get_height()):
                    if not self.map.is_wall((i,j)):
                        # Bellman equation
                        new_util = self.get_bellman((i,j))
                        self.map.set_utility((i, j), new_util)

            if self.has_converged(self.map, new_map):
                return
    
    def get_bellman(self, position):
        """
        Performs the bellman equation.

        :param position: current position as tuple
        :return: float. The new utility of the position
        """ 
        utilities = self.get_utilities_around(position)
        max_util = max(utilities.values())
        reward = self.map.get_reward(position)
        new_utility = reward + 0.3 * max_util
        return new_utility
    
    def has_converged(self, map1, map2):
        """
        Check if the utilities have converged 
        by comparing the new utility with the old utility

        :param map1: map with old utilities
        :param map2: map with new utilities
        :return: boolean
        """ 
        for x in range(self.map.get_width()):
            for y in range(self.map.get_height()):
                if map1.get_utility((x, y)) != map2.get_utility((x, y)):
                    return False
        return True
    
    def get_policy(self, position, legal):
        """
        Get the best policy.
        
        :param position: current position as tuple
        :return: the Direction that has the best utility
        """
        utilities = self.get_utilities_around(position)
        utilities = {key: utilities[key] for key in legal if key in utilities}
        return max(utilities, key=utilities.get)
    
    def get_utilities_around(self, position):
        """
        Get utilities of all surrounding neighbours

        :param position: current position as tuple
        :return: A map of all directions and their corresponding utilities
        """ 
        x, y = position
        north = (x, y + 1)
        south = (x, y - 1)
        east = (x + 1, y)
        west = (x - 1, y)

        utilities = {}
        north_util = self.map.get_neighbour_utility(north ,position)
        east_util = self.map.get_neighbour_utility(east, position)
        west_util = self.map.get_neighbour_utility(west, position)
        south_util = self.map.get_neighbour_utility(south, position)

        utilities[Directions.NORTH] = 0.8 * north_util + 0.1 * east_util + 0.1 * west_util
        utilities[Directions.SOUTH] = 0.8 * south_util + 0.1 * east_util + 0.1 * west_util
        utilities[Directions.EAST] = 0.8 * east_util + 0.1 * south_util + 0.1 * north_util   
        utilities[Directions.WEST] = 0.8 * west_util + 0.1 * south_util + 0.1 * north_util  

        return utilities
    
#########################################################
# get Action
#########################################################
    def getAction(self, state):
        legal = api.legalActions(state)
        pacman = api.whereAmI(state)

        #Update map
        self.clean_grid_reward()
        self.set_food(state)
        self.update_ghosts(state)

        # Value interation
        self.update_utilities()
        best_action = self.get_policy(pacman,legal)


        return api.makeMove(best_action, legal)

#########################################################
# Grid classes
#########################################################
class GridSquare:
    """
    A class that stores rewards, utilities and whether it is 
    a wall for each location in the grid.
    """
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.wall = False
        self.utility = 0
        self.reward = 0

class Grid:
    """
    An expanded version of Grid class from practical 5 by
    Mallmann-Trenn, Frederik

    A class that creates a grid that can be used as a map.
    The map itself is implemented as a nested list, and the interface
    allows it to be accessed by specifying x, y locations.
    """

#########################################################
# Original Grid class functions
#########################################################
    
    def __init__(self, width, height):
        """
        Constructor

        param grid:   an array that has one position for each element in the grid.
        param width:  the width of the grid
        param height: the height of the grid
        """
        self.width = width
        self.height = height

        # My version uses the class GridSquare to store values
        self.grid = [[GridSquare(i,j) for i in range(width)] for j in range(height)]

    def display(self):
        """
        Print the grid out
        """       
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[i][j].utility,
            # A new line after each line of the grid
            print 
        # A line after the grid
        print

    def prettyDisplay(self):    
        """
        The display function prints the grid out upside down. This
        prints the grid out so that it matches the view we see when we
        look at Pacman.
        """   
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[self.height - (i + 1)][j].reward,
            # A new line after each line of the grid
            print 
        # A line after the grid
        print

    def get_height(self):
        """
        Get the height of the grid

        return: height of grid
        """
        return self.height

    def get_width(self):
        """
        Get the width of the grid
        
        return: width of grid
        """
        return self.width
    
#########################################################
# New Grid functions
#########################################################
    def set_utility(self, xy, value):
        """
        Set square's utility

        param xy: indices of the map
        """
        x, y = xy
        self.grid[y][x].utility = value
    
    def get_utility(self, xy):
        """
        Get square's utility

        param xy: indices of the map
        return: utility
        """
        x, y = xy
        return self.grid[y][x].utility

    def get_neighbour_utility(self, neighbour, position):
        """
        Get the utility of neighbour square. If neighbour is a wall, 
        return the utility of the current position

        param neighbour: indices of neighbour
        param position: current position
        return: utility.
        """
        x, y = neighbour
        # If neighbour is a wall, return the current position's utility
        if self.grid[y][x].wall:
            i, j = position
            return self.grid[j][i].utility
        return self.grid[y][x].utility
    
    def set_reward(self, xy, value):
        """
        Set square's reward.

        param xy: indices of the map
        """
        x, y = xy
        self.grid[y][x].reward = value

    def get_reward(self, xy):
        """
        Get square's reward.

        param xy: indices of the map
        return: int
        """
        x, y = xy
        return self.grid[y][x].reward
    
    def set_wall(self, xy):
        """
        Set square as a wall.

        param xy: indices of the map
        """
        x, y = xy
        self.grid[y][x].wall = True
    
    def is_wall(self, xy):
        """
        Get whether the square is a wall.

        param xy: indices of the map
        return: boolean
        """
        x, y = xy
        return self.grid[y][x].wall