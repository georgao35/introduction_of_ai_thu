#ifndef Engine_H_
#define Engine_H_

#include <stdio.h>
#include <vector>
#include <unistd.h>
#include <time.h>
#include "Point.h"

#define usr 1
#define machine 2

#define TIME_LIMIT 2.6
#define maxn 14
#define r 10000

#define rival(x) (3-x)


struct Node {
    enum Status {NOT_TERMINATED, WON, TIE, LOST, UNKOWN};
    
    int m, n;
    int R, N;
    Status status;
    Point pos;
    int _player;
    int nxt_top_index;
    
    static int invalid_x, invalid_y;
    
    double c = 0.7;
    double gamma = 0.1;
    
    int board[maxn][maxn];
    int top[maxn];
    Node* par;
    std::vector<Node*> children;
    
    Node(Point pos, int player, int **board, const int *top, int m, int n);
    Node(Point pos, int player, int board[][maxn], const int *top, int m, int n);
    ~Node();
    
    Node* expand();
    Node* bestChild(int last=1);
    int defaultPolicy();
    bool terminal();
    
    void backup(int);
    
    int nearlyLose();
    int immediateWin();
};

class Engine {
    int m, n;
    Node* root;

public:
    Engine();
    Engine(const int m, const int n, int player, int **board, const int *top, Point last);
    void UCTS();
    ~Engine();
    Point getPoint();
    Node* treePolicy();
};

#endif /* Engine_hpp */
