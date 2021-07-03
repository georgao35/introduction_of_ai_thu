#include "Engine.h"
#include "Judge.h"
#include <cmath>

int Node::invalid_x = 0;
int Node::invalid_y = 0;

Node::Node(Point pos, int player, int **_board, const int *_top, int m, int n):pos(pos), m(m), n(n), _player(player), nxt_top_index(0), par(nullptr), R(0), N(1) {
    for (int i=0; i<n; i++) {
        top[i] = _top[i];
        for (int j=0; j<m; j++) {
            board[j][i] = _board[j][i];
        }
    }
    top[pos.y] = pos.x;
    if (pos.y==invalid_y && pos.x==invalid_x+1) {
        top[pos.y] = invalid_x;
    }
    if (pos.x>=0 && pos.x<m && pos.y>=0 && pos.y<n) {
        board[pos.x][pos.y] = player;
    }
    status = UNKOWN;
}

Node::Node(Point pos, int player, int _board[][maxn], const int *_top, int m, int n):pos(pos), m(m), n(n), _player(player), nxt_top_index(0),par(nullptr), R(0), N(1) {
    for (int i=0; i<n; i++) {
        top[i] = _top[i];
        for (int j=0; j<m; ++j) {
            board[j][i] = _board[j][i];
        }
    }
    top[pos.y] = pos.x;
    if (pos.y==invalid_y && pos.x==invalid_x+1) {
        top[pos.y] = invalid_x;
    }
    if (pos.x>=0 && pos.x<m && pos.y>=0 && pos.y<n) {
        board[pos.x][pos.y] = player;
    }
    status = UNKOWN;
}

Node::~Node() {
    // 怎么析构
    for (Node *p : children) {
        if (p) {
            delete p;
            p = nullptr;
        }
    }
}

Node* Node::expand() {
    while (nxt_top_index<n && top[nxt_top_index]==0)
        ++nxt_top_index;
    if (nxt_top_index>=n) {
        return nullptr;
    }
    Point new_pos(top[nxt_top_index]-1, nxt_top_index);
    Node *node = new Node(new_pos, rival(_player), board, top, m, n);
    ++nxt_top_index;
    node->par = this;
    children.push_back(node);
    return node;
}

Node* Node::bestChild(int last) {
    Node *best = nullptr;
    double max_reward = (double)(0-0xffffff);
    for (Node *p : children) {
        if (p->immediateWin())
            return p;
    }
    for (Node* i: children) {
        double reward = (double) i->R / (double) i->N;
        reward += last * c * sqrt( 2*log(N+1.001) / (double)i->N);
        int total_rival = i->nearlyLose();
        if (total_rival==-1) return i;
        reward += (double)total_rival / log(N+1.01) * 0.1;
        if (reward > max_reward) {
            max_reward = reward;
            best = i;
        }
    }
    return best;
}

bool Node::terminal() {
    if (status!=UNKOWN){
        return status;
    }
    if (pos.x < 0 || pos.x >= m) return false;
    if (pos.y < 0 || pos.y >= n) return false;
    int *state[maxn];
    for (int i=0; i<m; i++) {
        state[i] = new int [n];
        for (int j=0; j<n; j++)
            state[i][j] = board[i][j];
    }
    if (_player==usr && userWin(pos.x, pos.y, m, n, state))
        status = WON;
    else if (_player==machine && machineWin(pos.x, pos.y, m, n, state))
        status = WON;
    else if (isTie(n, top)) status = TIE;
    else status = NOT_TERMINATED;
    for (int i=0; i<m; i++) {
        delete [] state[i];
    }
    return status & 1;
}

int Node::defaultPolicy() {
    /* 随机模拟当前棋局至分出胜负 */
    if (terminal()) {
        if (status == Node::Status::WON) {
            return r;
        } else if (status == Node::Status::LOST) {
            return 0-r;
        }
        else return 0;
    }
    int *board_policy[maxn];
    int top_policy[maxn];
    for (int j=0; j<n; j++) {
        top_policy[j] = top[j];
    }
    for (int i=0; i<m; i++) {
        board_policy[i] = new int [n];
        for (int j=0; j<n; j++) {
            board_policy[i][j] = board[i][j];
        }
    }
    int now_player = _player;
    int x, y;
    int delta = 0;
    double co = r;
    while (true) {
        int i;
        do {
            i = random() % n;
        }while (top_policy[i]==0);
        x = top_policy[i]-1; y = i;
        now_player = rival(now_player);
        board_policy[x][y] = now_player;
        top_policy[y] = x;
        if (x-1==invalid_x && y==invalid_y && x>0)
            top_policy[y]--;
        if (now_player==usr && userWin(x, y, m, n, board_policy)) {
            delta = _player==usr? 1:-1;
            break;
        } else if (now_player==machine && machineWin(x, y, m, n, board_policy)) {
            delta = _player==machine? 1:-1;
            break;
        } else if (isTie(n, top_policy)) {
            delta = 0;
            break;
        }
        co *= gamma;
    }
    for (int i=0; i<m; i++) {
        delete [] board_policy[i];
    }
    return delta;
}

void Node::backup(int delta) {
    Node *p = this;
    do {
        p->R+= delta;
        p->N++;
        delta *= -1;
        if (abs(delta) > 1){
            int abs_delta = abs(delta)*gamma;
            if (abs_delta < 1) abs_delta = 1;
            delta = delta>0? abs_delta:0-abs_delta;
        }
        p = p->par;
    } while (p);
}

int Node::nearlyLose() {
    if (pos.x < 0 || pos.x >= m
        || pos.y < 0 || pos.y >= n)
        return 0;
    int count = 0, total = 0;
    //竖向
    for (int i=pos.x + 1; i<m; i++) {
        if (board[i][pos.y] == rival(_player)) count++;
        else break;
    }
    if (count >= 3) return -1;
    total += count;
    //横向
    count = 0;
    for (int i=pos.y-1; i>=0; i--) {
        if (board[pos.x][i] == rival(_player)) count++;
        else break;
    }
    for (int i=pos.y+1; i<n; i++) {
        if (board[pos.x][i] == rival(_player)) count++;
        else break;
    }
    if (count >= 3) return -1;
    total += count;
    //左上-右下
    count = 0;
    int i, j;
    for (i = pos.x - 1, j = pos.y - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == rival(_player)) count++;
        else break;
    }
    for (i = pos.x + 1, j = pos.y + 1; i < m && j < n; i++, j++) {
        if (board[i][j] == rival(_player)) count++;
        else break;
    }
    if (count >= 3) return -1;
    total += count;
    //右上-左下
    count = 0;
    for (i = pos.x + 1, j = pos.y - 1; i < m && j >= 0; i++, j--) {
        if (board[i][j] == rival(_player)) count++;
        else break;
    }
    for (i = pos.x - 1, j = pos.y + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == rival(_player)) count++;
        else break;
    }
    total += count;
    if (count >= 3) return -1;
    return total;
}

int Node::immediateWin() {
    if (pos.x < 0 || pos.x >= m
        || pos.y < 0 || pos.y >= n)
        return 0;
    int count = 0;
    //竖向
    for (int i=pos.x + 1; i<m; i++) {
        if (board[i][pos.y] == _player) count++;
        else break;
    }
    if (count >= 3) return 1;
    //横向
    count = 0;
    for (int i=pos.y-1; i>=0; i--) {
        if (board[pos.x][i] == _player) count++;
        else break;
    }
    for (int i=pos.y+1; i<n; i++) {
        if (board[pos.x][i] == _player) count++;
        else break;
    }
    if (count >= 3) return 1;
    //左上-右下
    count = 0;
    int i, j;
    for (i = pos.x - 1, j = pos.y - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == _player) count++;
        else break;
    }
    for (i = pos.x + 1, j = pos.y + 1; i < m && j < n; i++, j++) {
        if (board[i][j] == _player) count++;
        else break;
    }
    if (count >= 3) return 1;
    //右上-左下
    count = 0;
    for (i = pos.x + 1, j = pos.y - 1; i < m && j >= 0; i++, j--) {
        if (board[i][j] == _player) count++;
        else break;
    }
    for (i = pos.x - 1, j = pos.y + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == _player) count++;
        else break;
    }
    if (count >= 3) return 1;
    return 0;
}

Engine::Engine(const int m, const int n, int player, int **board, const int *top, Point last):m(m), n(n) {
//    printf("Engine\n");
    root = new Node(last, player, board, top, m, n);
}

Engine::~Engine() {
    if (root) {
        delete root;
        root = nullptr;
    }
}

Node* Engine::treePolicy() {
//    printf("treePoicy\n");
    Node *now = root;
    while (now && !now->terminal()) {
        Node *tmp = now->expand();
        if (tmp)
            return tmp;
        else
            now = now->bestChild();
    }
    return now;
}

void Engine::UCTS() {
    clock_t start = clock();
    while (clock()-start<= TIME_LIMIT * CLOCKS_PER_SEC) {
        Node *node = treePolicy();
        int delta = node->defaultPolicy();
        node->backup(delta);
    }
}

Point Engine::getPoint(){
    UCTS();
    return root->bestChild(0)->pos;
}
