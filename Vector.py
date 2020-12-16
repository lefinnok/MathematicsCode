from wincon.WinCon import Screen, Float, Frame, CommandLine, Label, TextBox
from Matrix import Matrix, summ, sub, T


'઻'


class Vector(Matrix):
    def __init__(self, vector):
        super().__init__([vector])

class DDCoordinateSystem():
    def update(self):

        maxC = (max([abs(self.coordinates[cord][0]) for cord in self.coordinates]),max([abs(self.coordinates[cord][1]) for cord in self.coordinates]))
        empty = [['઻' if x != maxC[0]*2 else '|' for x in range(maxC[0]*4+1) ] if y != maxC[1] else ['-' for x in range(maxC[0]*4+1)] for y in range(maxC[1]*2+1) ]



        for connection in self.connections: #Draw Connections
            point1 = self.coordinates[connection[0]]
            point2 = self.coordinates[connection[1]]
            if point1[0] > point2[0]:
                dx = point1[0]-point2[0]
                dy = point1[1]-point2[1]
                for x in range(point2[0],point1[0]):
                    y = -round(point2[1] + dy * (x-point2[0]) / dx)
                    empty[(y+maxC[1])][(x+maxC[0])*2] = '#'
            else:
                dx = point2[0]-point1[0]
                dy = point2[1]-point1[1]
                for x in range(point1[0],point2[0]):
                    y = -round(point1[1] + dy * (x-point1[0]) / dx)
                    empty[(y+maxC[1])][(x+maxC[0])*2] = '#'

        for point in self.coordinates: #Draw Points
            coord = self.coordinates[point]
            '''
            print((coord[0]+maxC[0]),(coord[1]+maxC[1])*2)
            print(len(empty[0]),len(empty))
            '''
            empty[-(coord[1]+maxC[1])-1][(coord[0]+maxC[0])*2] = point


        self.float = Float(empty, 0, 0, True)
        self.frame.floats = [self.float]

    def __init__(self, coordinates, screen, connections=[]):
        self.coordinates = coordinates
        self.screen = screen
        self.frame = self.screen.floats[0]
        self.connections = connections
        self.vectors = {}
        self.update()

    def addPoint(self,char,x,y):
        self.coordinates[char[0]] = (int(x),int(y))

    def addConnection(self,A,B):
        if A in self.coordinates and B in self.coordinates:
            self.connections.append((A,B))
            '''
            This adds a vector to the dict of vectors
            for coordinate of point A, is OA
            and coordinate of point B. is OB

            to get vector AB:
            AB = AO + OB = -OA + OB = OB - OA
            '''
            try:
                self.vectors[A+B] = sub(Vector(list(self.coordinates[B])),Vector(list(self.coordinates[A])))
            except Exception as e:
                print(e)
                screen.Ex=True


    def removePoint(self,char):
        self.coordinates.pop(char)
        for connection in self.connections:
            if char in connection:
                self.connections.remove(connection)
        for vector in self.vectors:
            if char in vector:
                self.vectors.pop(vector)

    def removeConnection(self,A,B):
        if (A,B) in self.connections:
            self.connections.remove((A,B))
        if (A + B) in self.vectors:
            self.vectors.pop(A+B)
            
    def listVectors(self):
        for vector in self.vectors:
            prompt(f'{vector}:')
            prompt(self.vectors[vector])




def prompt(text):
    try:
        text = str(text).split('\n')
        for t in text:
            textbox.addText([text])
    except Exception as e:
        print(e)
        screen.Ex=True


def addPoint(char,x,y):
    try:
        coordSys.addPoint(char,x,y)
    except Exception as e:
        print(e)
        screen.Ex=True
    prompt(f'Point {char} added at ({x},{y})')
    coordSys.update()

def addConnection(A,B):
    try:
        coordSys.addConnection(A,B)

    except Exception as e:
        print(e)
        screen.Ex=True
    prompt(f'Point {A} and Point {B} are connected')
    coordSys.update()

def removePoint(char):
    try:
        coordSys.removePoint(char)
    except Exception as e:
        print(e)
        screen.Ex=True
    prompt(f'Point {char} removed')
    coordSys.update()

def removeConnection(A,B):
    try:coordSys.removeConnection(A,B)
    except Exception as e:
        print(e)
        screen.Ex=True
    prompt(f'Point {A} and Point {B} are disconnected')
    coordSys.update()

def listVectors():
    coordSys.listVectors()

textList = []

screen = Screen(fps=60, fullscreen=False, scrollSensitivity=2)

frame = Frame([],113,63,0,0,True,screen)

command = CommandLine(114,113,60,True,{'add': addPoint, 'del': removePoint, 'connect': addConnection, 'disconnect': removeConnection, 'vectors': listVectors})

textbox = TextBox(114,61,113,0,True, textList)

screen.addFloats([frame,command,textbox])

coordSys = DDCoordinateSystem({'0': (0,0)},screen,[])
screen.start()
