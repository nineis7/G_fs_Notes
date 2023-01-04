#include<stdio.h>
#include<stdlib.h>
#define len sizeof(struct xs)

struct xs
{
    int xh;
    char xm[100];
    int gs;
    int dw;
    int dy;
    int c;
    struct xs*next;
};
struct xs *create()
{
	struct xs *head=NULL;
	struct xs *n;
	struct xs *tail;
	int count=0;
	while(1)
	{
		n=(struct xs *)malloc(len);
		printf("请输入学号：\n");
		scanf("%d",&n->xh);
		if(n->xh==0)
		{
			free(n);
			break;
		}
		printf("请输入姓名：\n");
		scanf("%s", n->xm);
		getchar();
		printf("请输入高等数学、大学物理、英语和 C 语言四门课程的成绩：\n");
		scanf("%d,%d,%d,%d",&n->gs,&n->dw,&n->dy,&n->c);
		getchar();
		
		if(count==0)
		{
			head=n;
			tail=n;
		}
		else
		{
			tail->next=n;
			tail=n;
		}
		count++;
	}
	tail->next=NULL;
	return head;
}
void gkjf(struct xs *head)
{
	struct xs *s;
	s=head;
	int i=0,gszf=0,gsjf,dwzf=0,dwjf,dyzf=0,dyjf,czf=0,cjf;
	if(s==NULL)
	   printf("这是个空表");
	else
	{
		while(s!=NULL)
		{
			printf("gszf=%d, i=%d\n", gszf, i);
			gszf=gszf+s->gs;
			dwzf=dwzf+s->dw;
			dyzf=dyzf+s->dy;
			czf=czf+s->c;
			i++;
			s=s->next;
		}
		printf("gszf=%d, i=%d\n", gszf, i);
		gsjf=gszf/i;
		dwjf=dwzf/i;
		dyjf=dyzf/i;
		cjf=czf/i;
		printf("高数均分为%d,大物均分为%d，英语均分为%d，C语言均分为%d",gsjf,dwjf,dyjf,cjf); 
	}
	    
}

int main()
{
	struct xs *head;
	head=create();
	printf("xh=%d, dw=%d\n", head->xh, head->dw);
	gkjf(head);
}
