/**
 * Simple program to count lines for the program
 *
 * Usage:
 *
 * ./count_lines <dirpath>
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct file
{
    char *name;
    int lines;
    struct file *left;
    struct file *right;
};

static struct file *head = NULL;
static int max = 0;
static int total = 0;
static int total_files = 0;
static int num_of_stars = 20;

struct file *Find(struct file *h, int lines)
{

    if (h == NULL)
    {
        return h;
    }
    else if (h->lines < lines)
    {
        if (h->right == NULL)
        {
            return h;
        }
        else
        {
            return Find(h->right, lines);
        }
    }
    else
    {
        if (h->left == NULL)
        {
            return h;
        }
        else
        {
            return Find(h->left, lines);
        }
    }
}

void InsertFile(char *name, int lines)
{

    if (lines > max)
        max = lines;
    total += lines;
    total_files += 1;

    struct file *new = (struct file *)malloc(sizeof(struct file));

    // Allocate a new node
    new->name = (char *)malloc(sizeof(char) * (strlen(name) + 1));

    strcpy(new->name, name);

    new->lines = lines;
    new->left = NULL;
    new->right = NULL;

    struct file *f = Find(head, lines);

    if (f == NULL)
    {
        head = new;
    }
    else if (f->lines < lines)
    {
        f->right = new;
    }
    else
    {
        f->left = new;
    }
}

void PrintFiles(struct file *f)
{

    if (f->right != NULL)
    {
        PrintFiles(f->right);
    }

    float scale = (float)f->lines / (1.0f * max);
    int num = (scale * (num_of_stars - 1));

    for (int i = 0; i < num_of_stars; i++)
    {
        if (i > num)
            printf(" ");
        else
            printf("*");
    }

    printf(" | %4d | %s", f->lines, f->name);

    puts("");

    if (f->left != NULL)
    {
        PrintFiles(f->left);
    }
}

// Count non-blank lines in a file
int CountLines(char *filename)
{
    FILE *fp = fopen(filename, "r");
    int count = 0;
    char c;

    if (fp == NULL)
    {
        printf("Could not open file %s", filename);
        return 0;
    }

    for (c = getc(fp); c != EOF; c = getc(fp))
    {
        if (c == '\n')
        {
            count++;
        }
    }

    fclose(fp);
    return count;
}

void ListFiles(char *BasePath, int Level)
{

    char path[1000];
    struct dirent *dp;
    DIR *dir = opendir(BasePath);

    // Unable to open directory stream
    if (!dir)
        return;

    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {

            // Construct new path from our base path
            strcpy(path, BasePath);
            strcat(path, "/");
            strcat(path, dp->d_name);

            switch (dp->d_type)
            {
            case DT_REG:

                // Check if file is a .c, .h or .py file
                if (strstr(dp->d_name, ".c") == NULL && strstr(dp->d_name, ".h") == NULL &&
                    strstr(dp->d_name, ".py") == NULL)
                    break;

                // Get file size with stat
                int clines = CountLines(path);

                InsertFile(dp->d_name, clines);
                break;
            case DT_DIR:
                ListFiles(path, Level + 1);
                break;
            default:
                printf("(o)");
                puts("");
                break;
            }
        }
    }

    closedir(dir);
}

int main(int argc, char **argv)
{
    // Specify the path you want to start from
    char path[100];

    if (argc == 1)
    {
        printf("Usage: %s <path>\n", argv[0]);
        return 1;
    }
    else
    {
        strcpy(path, argv[1]);
    }

    printf("Counting lines for %s\n", path);

    ListFiles(path, 0);

    printf("Total files: %d\n", total_files);
    printf("Total lines: %d\n", total);

    puts("");

    PrintFiles(head);

    return 0;
}
