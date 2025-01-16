#include <stdio.h>
#include <stdlib.h>

#define MAX_N 100000

int heights[MAX_N];
int stack[MAX_N];
int maxAreaSoFar;

int getMaxArea(int i, int n) {
    int maxArea = 0;
    int top = -1;
    for (int j = i; j < n; j++) {
        while (top != -1 && heights[stack[top]] > heights[j]) {
            int height = heights[stack[top--]];
            int width = top == -1 ? j : j - stack[top] - 1;
            maxArea = (maxArea > height * width) ? maxArea : height * width;
        }
        stack[++top] = j; 
    }

    while (top != -1) {
        int height = heights[stack[top--]];
        int width = top == -1 ? n : n - stack[top] - 1;
        maxArea = (maxArea > height * width) ? maxArea : height * width;
    }

    return maxArea;
}

int main() {
    int n;
    while (scanf("%d", &n) && n) {
        for (int i = 0; i < n; ++i) {
            scanf("%d", &heights[i]);
        }
        maxAreaSoFar = 0;
        maxAreaSoFar = getMaxArea(0, n);
        printf("%d\n", maxAreaSoFar);
    }
    return 0;
}