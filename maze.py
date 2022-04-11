import cv2
import threading
import time

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


click_status = 0

rw = 4 # for position of the rectangle drawn on image when clicked

start = Point()  # starting point
end = Point()  # ending point

# directions of surrounding nodes for BFS
direction_points = [Point(0, -1), Point(0, 1), Point(1, 0), Point(-1, 0)]



#==========================================================================================
#=====priority queue==========
def heapify(arr,i):
	tiniest=i
	l=2*i+1
	r=2*i+2
	if tiniest<len(arr):
		a,b=arr[tiniest]
	if l<len(arr):
		f,g=arr[l]
	if r<len(arr):
		h,j=arr[r]
	if l<len(arr) and b>g:
		tiniest = l
		a,b=arr[tiniest]
	if r<len(arr) and b>j:
		tiniest = r
		a,b=arr[tiniest]
	if tiniest!=i:
		t = arr[tiniest]
		arr[tiniest] = arr[i]
		arr[i] = t
		heapify(arr,tiniest)

class priority_queue:
	def __init__(self):
		self.cap=-1
		self.pq=[]
	def size(self):
		return len(self.pq)
	def push(self,s,d):
		self.pq.append((s,d))
		self.cap+=1
		i=self.cap
		a,b=self.pq[int((i-1)/2)]
		while i!=0 and b>d:
			t=self.pq[i]
			self.pq[i]=self.pq[int((i-1)/2)]
			self.pq[int((i-1)/2)]=t
			a,b=self.pq[i]
			i=int((i-1)/2)
	def _pop(self):
		if(len(self.pq)>0):
			t=self.pq[0]
			self.pq[0]=self.pq[self.cap]
			heapify(self.pq,0)
			self.cap-=1
			self.pq.pop()
			return t


#=============================================================================A* stuff==================
#===================================
def isSafe(temp_x,temp_y,h,w,img,visited):
	return temp_x>=0 and temp_x<w and temp_y>=0 and temp_y<h and visited[temp_y][temp_x]==0 and (img[temp_y][temp_x][0]!=0 or img[temp_y][temp_x][1]!=0 or img[temp_y][temp_x][2]!=0)

def heuristic_cost(s,e):
	return abs(s.x - e.x) + abs(s.y - e.y)

a_star_pixel_count=0

def a_star(s,e,h,w,img):
        global start_time
        pq=priority_queue()
        const=2000
        found=0
        visited_tracker = [[0 for i in range(w)]for j in range(h)]
        path_tracker = [[Point() for i in range(w)]for j in range(h)]
        d = heuristic_cost(s,e)
        pq.push(s,d)
        x_c=[0,0,1,-1]
        y_c=[1,-1,0,0]
        visited_tracker[s.y][s.x]=1
        a_star_pixel_count=0
        while pq.size()>0:
            temp, g = pq._pop()
            if(temp == e):
                found=1
                break
            for i in range(4):
                temp_x = temp.x + x_c[i]
                temp_y = temp.y + y_c[i]
                if isSafe(temp_x,temp_y,h,w,img,visited_tracker) == True:
                    visited_tracker[temp_y][temp_x]=visited_tracker[temp.y][temp.x]+1
                    path_tracker[temp_y][temp_x]=temp
                    d = heuristic_cost(Point(temp_x,temp_y),e)
                    pq.push(Point(temp_x,temp_y),d)
                    a_star_pixel_count+=1
                    img[temp_y][temp_x]=[33,33,84]#list(reversed([i * 255 for i in colorsys.hsv_to_rgb(visited[t_y][t_x] / const, 1, 1)]))

        path=[]
        pixels_in_path=0
        if found==1:
                i=e
                while i!=s:
                    path.append(Point(i.x,i.y))
                    i=path_tracker[i.y][i.x]
                path.append(s)
                path.reverse()
                for p in path:
                    for i in range(-1,1):
                        pixels_in_path+=1
                        img[p.y+i][p.x+i]=[255,0,0]
                print("A Star")
                print("Number of pixels visited to find a path: ", a_star_pixel_count)
                print("Length of path: %s pixels" % pixels_in_path)
                print("Execution time: %s seconds" % (time.time() - start_time))
                print("========================================")

        else:
                print("Path not found")
#=====================================================


# The BFS function:
# All the pixels surrounding the starting pixel acts as neighbouring nodes
# Breadth First Search is performed until the ending point is found
def bfs(s, e):
    print("========================================")
    global img, h, w, start_time

    # ending point found
    found = False

    # queue to perform BFS
    queue = []

    # visited matrix of size width*height of the image with each element = 0
    visited_tracker = [[0 for j in range(w)] for i in range(h)]
    # parent matrix of size width*height of the image with each element = empty Point object
    path_tracker = [[Point() for j in range(w)] for i in range(h)]

    # storing starting Point and marking it 1 in visited matrix
    queue.append(s)
    visited_tracker[s.y][s.x] = 1
    total_bfs_points = 0
    # looping until queue is not empty
    while len(queue) > 0:
        # popping one element from queue and storing in p
        p = queue.pop(0)
        
        # surrounding elements
        for d in direction_points:
            cell = p + d
          
            # if cell(a surrounding pixel) is in range of image, not visited, !(B==0 && G==0 && R==0) i.e. pixel is
            # not black as black represents border
            if (0 <= cell.x < w and 0 <= cell.y < h and
            visited_tracker[cell.y][cell.x] == 0 and
                    (img[cell.y][cell.x][0] != 0 or img[cell.y][cell.x][1] != 0 or img[cell.y][cell.x][2] != 0)):
                total_bfs_points= total_bfs_points + 1
                queue.append(cell)

                # marking cell as visited
                visited_tracker[cell.y][cell.x] = 1
                # changing the pixel color of flood
                img[cell.y][cell.x] = [80, 80, 80]

                # string the value of p in parent matrix to trace path
                path_tracker[cell.y][cell.x] = p

                # if end is found break
                if cell == e:
                    found = True
                    del queue[:]
                    break

    # list to trace path
    # the parent is a matrix list of objects. p=e is key here. it means that we are tracing back to the start point from the end point. It works because of the way parent was populated in the loop before. The correct path will start from the end point since parent[cell.y][cell.x] = p is done before.
    pixels_in_path = 0
    path = []
    if found:
        p = e

        while p != s:
            path.append(p)
            p = path_tracker[p.y][p.x]

        path.append(p)
        path.reverse()

        # changing the pixel of resulting path 
        for p in path:
            pixels_in_path+=1
            img[p.y][p.x] = [0, 0, 255]
            img[p.y+1][p.x+1] = [0, 0, 255]
            # img[p.y-1][p.x-1] = [0, 0, 255]
    print("BFS")
    print("Number of pixels visited to find a path: ",total_bfs_points)
    print("Length of path: %s pixels" % pixels_in_path)
    print("Exection time: %s seconds" % (time.time() - start_time))
    print("========================================")

# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
def mouse_click(event, px, py, flag, params):

    global img, click_status, start, end, rw

    # if left clicked
    if event == cv2.EVENT_LBUTTONUP:
        # first click
        if click_status == 0:
            # create a rectangle on the image
            cv2.rectangle(img, (px-rw, py-rw), (px+rw+1, py+rw+1), (0, 0, 255), -1)
            # store the starting point
            start = Point(px, py)
            # change status to 1
            click_status = click_status + 1

        # second click
        elif click_status == 1:
            # create a rectangle on the image
            cv2.rectangle(img, (px-rw, py-rw), (px+rw+2, py+rw+2), (0, 255, 0), -1)
            # store the ending point
            end = Point(px, py)
            # change status to 2
            click_status = click_status + 1
# =============================================================
start_time = time.time()
# display function
def calculations():
    global start_time
    # Preventing BFS function to run until both clicks are performed and the starting and ending point are stored
    global click_status,start, end, h, w
    while click_status < 2:
        pass
    
    # calling BFS function
    start_time = time.time()
    bfs(start, end)

    # calling A_star function
    start_time = time.time()
    a_star(start,end,h,w,img)


# reading the image
img = cv2.imread("./theseus.jpg")#PLEASE ADD THE ABSOLUTE PATH OF THE MAZE IMAGE IF YOU RUN THIS CODE
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (1080, 720), interpolation=cv2.INTER_NEAREST)
_, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# img object, threshold value, color of threshold, Binary thresh because 2 colors

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# height and width of image
h, w = img.shape[:2]

# thread to call bfs in parallel

t = threading.Thread(target=calculations, args=())
t.daemon = True
t.start()

cv2.imshow("image", img)
cv2.setMouseCallback("image", mouse_click)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        break

