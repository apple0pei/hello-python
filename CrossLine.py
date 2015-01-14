class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

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
        if x_0 < min(line1.p1.x,line1.p2.x) or x_0 < min(line2.p1.x,line2.p2.x) or x_0 > max(line1.p1.x,line1.p2.x) or x_0 > max(line2.p1.x,line2.p2.x) or y_0 < min(line1.p1.y,line1.p2.y) or y_0 < min(line2.p1.y,line2.p2.y) or y_0 > max(line1.p1.y,line1.p2.y) or y_0 > max(line2.p1.y,line2.p2.y):
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
    line1 = Line(Point(-5,-5), Point(5,5))
    line2 = Line(Point(5,-5), Point(-5,5))
    assert lineCross(line1, line2)==1
    