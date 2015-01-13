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
    else:
        x_0 = (beta_2-beta_1) / (alpha_1-alpha_2)
        y_0 = (alpha_1*beta_2-alpha_2*beta_1) / (alpha_1-alpha_2)
        if x_0 < min(line1.p1.x,line1.p2.x) or x_0 < min(line2.p1.x,line2.p2.x) or x_0 > max(line1.p1.x,line1.p2.x) or x_0 > max(line2.p1.x,line2.p2.x) or y_0 < min(line1.p1.y,line1.p2.y) or y_0 < min(line2.p1.y,line2.p2.y) or y_0 > max(line1.p1.y,line1.p2.y) or y_0 > max(line2.p1.y,line2.p2.y):
                print "Two lines do not cross on each other because x=%s and y=%s is out of the range!" % (x_0,y_0)
        else:
            print "Two lines cross at point x=%s and y=%s" % (x_0,y_0)
           

# def lineCrossUgly(x1, y1, x2, y2, x3, y3, x4, y4):
    

if __name__ == '__main__':
    line1 = Line(Point(0,0), Point(4,4))
    line2 = Line(Point(0,1), Point(5,-1))
    lineCross(line1, line2)

    
    