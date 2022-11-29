#include<vector>
#include<stdlib.h>

using namespace std;


vector<pair<int, int>>
FindDirectedTrees(const int** graph, const int n, const int begin_node)
{
    vector<pair<int, int>> ret_vec;
    vector<int> _visited(n, 0);
    int i(0);
    int row(begin_node);
    _visited[begin_node] = 1;

    while(row < n){
        if(_visited[row] == 1){
            //从指定结点遍历其他未访问结点
            for(int col(0); col < n; ++col){
                if(graph[row][col] && !_visited[col]) {
                    _visited[col] == 1;
                    ret_vec.push_back(make_pair(row,col));
                }
            }
            _visited[row] = 2;
            row = 0;
        }
        else row++;
        
    }

    return ret_vec;
}
