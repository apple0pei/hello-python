import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def isInLine(self, x1, y1):
        if x1 < min(self.p1.x,self.p2.x) or x1 > max(self.p1.x,self.p2.x) or y1 < min(self.p1.y,self.p2.y) or y1 > max(self.p1.y,self.p2.y):
            return False
        else:
            return True


def lineCross(line1, line2):
    # y_1 = alpha_1 * x_1 + beta_1 (y_1, x_1 points on line1 bound by p1 and p2 on line1)
    alpha_1 = 1.0 * (line1.p2.y-line1.p1.y) / (line1.p2.x-line1.p1.x)
    beta_1 = 1.0 * (line1.p2.x*line1.p1.y - line1.p1.x*line1.p2.y) / (line1.p2.x-line1.p1.x)

    alpha_2 = 1.0 * (line2.p2.y-line2.p1.y) / (line2.p2.x-line2.p1.x)
    beta_2 = 1.0 * (line2.p2.x*line2.p1.y - line2.p1.x*line2.p2.y) / (line2.p2.x-line2.p1.x)
    
    if alpha_1 == alpha_2:
        print "Two lines are parallel to each other!"
        return 0
    else:
        x_0 = (beta_2-beta_1) / (alpha_1-alpha_2)
        y_0 = (alpha_1*beta_2-alpha_2*beta_1) / (alpha_1-alpha_2)
        if line1.isInLine(x_0,y_0)==False or line2.isInLine(x_0,y_0)==False:
            print "Two lines do not cross on each other because x=%s and y=%s is out of the range!" % (x_0,y_0)
            return 0        
        else:
            print "Two lines cross at point x=%s and y=%s" % (x_0,y_0)
            return 1

# def lineCrossUgly(x1, y1, x2, y2, x3, y3, x4, y4):
    

if __name__ == '__main__':
#test1: parallel lines
    line1 = Line(Point(0,0), Point(5,5))
    line2 = Line(Point(1,0), Point(6,5))
    assert lineCross(line1, line2)==0

#test2: not across
    line1 = Line(Point(0,0), Point(5,5))
    line2 = Line(Point(-1,-3), Point(-8,-6))
    assert lineCross(line1, line2)==0
    
#test3: across
    line1 = Line(Point(-4,-7), Point(5,6))
    line2 = Line(Point(5,-5), Point(-5,5))
    assert lineCross(line1, line2)==1

fig = plt.figure()
ax = fig.add_subplot(111)
line1_x = [line1.p1.x,line1.p2.x] 
line1_y = [line1.p1.y,line1.p2.y] 
line = ax.plot(line1_x, line1_y, 'bs-', picker=5)

line2_x = [line2.p1.x,line2.p2.x] 
line2_y = [line2.p1.y,line2.p2.y] 
line = ax.plot(line2_x, line2_y, 'bs-', picker=5)

plt.show()
    