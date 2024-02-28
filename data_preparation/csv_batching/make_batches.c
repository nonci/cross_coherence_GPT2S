/* gcc -Wall make_batches.c -o make_batches
   python3 sort.py chatgpt_text2shape.csv chatgpt_text2shape_sorted.csv
   ./make_batches chatgpt_text2shape_sorted.csv chatgpt_text2shape2.csv $(cat chatgpt_text2shape.csv| wc --lines) 1024
    CSV file is assumed to be comma-separated with the id in 2nd position.
    The 1st line is ignored.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_ID_REPETITIONS 20

int main(int argc, char **argv) {
    FILE * csv = fopen(argv[1], "r");
    if (!csv) exit(1); 
    FILE * csv_out = fopen(argv[2], "a");
    if (!csv_out) exit(1);

    unsigned n_items = atoi(argv[3])-1;
    unsigned BATCHSIZE = atoi(argv[4]);
    unsigned first_free_batch = 0;
    unsigned n_batches = n_items/BATCHSIZE+1+MAX_ID_REPETITIONS; //if division is exact we'll have 1 extra batch

    typedef struct {
        char * lines[BATCHSIZE];
        unsigned free_pos;
    } batch;
    batch batches[n_batches]; 

    char * line = NULL;
    char previous_id[100] = {0}, curr_id[100];
    size_t n_read;
    unsigned i,j;
    unsigned curr_batch;

    printf("items: %d\n", n_items);

    bzero((void*)batches, n_batches*sizeof(batch));

    // the first line is directly written on out.file:   //fscanf(csv, "%*s\n");
    getline(&line, &n_read, csv);
    fwrite(line, strlen(line), 1, csv_out);

    while (getline(&line, &n_read, csv)!=EOF) {
        if (line[0]=='\n') continue;

        for (i=0; i<n_read; i++)
            if (line[i]==',') break;

        for (j=i+i; j<n_read; j++)
            if (line[j]==',') break;
        
        strncpy(curr_id, &line[i+1], j-i-1);
        curr_id[j-i-1] = '\0';
        if (strcmp(previous_id, curr_id) == 0)
            curr_batch += 1;
        else
            curr_batch = first_free_batch;

        strcpy(previous_id, curr_id);        

        batches[curr_batch].lines[batches[curr_batch].free_pos] = line, line = NULL;

        batches[curr_batch].free_pos++;
        if (batches[curr_batch].free_pos == BATCHSIZE)
            first_free_batch++;
        if (first_free_batch == n_batches){
            printf("Out of batches!\n");
            exit(2);
        }

    }
    //printf("FULL BATCHES: %d\n", first_free_batch);

    /*** SWAPPING STARTS ***/
    // get 1st not-full batch index in curr_batch:
    for (curr_batch = n_batches-1; batches[curr_batch].free_pos!=BATCHSIZE; curr_batch--)
        ;
    curr_batch++;
    
    unsigned swap_end_pos_b = n_batches-1;
    // get 1st not-empty batch index in swap_end_pos_b:
    for (; batches[swap_end_pos_b].free_pos==0; swap_end_pos_b--)
        ;

    //char * tmp_line = NULL;
    unsigned swap_start_l = 0;
    unsigned b_change_needed;
    for (int swap_pos_b=0, swap_pos_l=0; (swap_start_l<BATCHSIZE) && swap_end_pos_b>curr_batch; ) { //swap_pos_l<BATCHSIZE
        batches[curr_batch].lines[batches[curr_batch].free_pos] = batches[swap_pos_b].lines[swap_pos_l];
        batches[swap_pos_b].lines[swap_pos_l] = batches[swap_end_pos_b].lines[batches[swap_end_pos_b].free_pos-1];
        b_change_needed=0;

        //update free_pos and, if necessary, curr_batch: (1/2)
        if ( --(batches[swap_end_pos_b].free_pos) == 0 ){
            swap_end_pos_b--;
            // than the swap_pos_b has to be changed too, since we don't want duplicates in batches[swap_pos_b]
            b_change_needed = 1;
        } 
        
        if (swap_pos_b==curr_batch-MAX_ID_REPETITIONS-1) { // TODO: test branch
             // update swap position due to safe bottom reached:
            puts("Bottom reach; trying restart!"),
            swap_pos_b=0, swap_pos_l=++swap_start_l;
        } else if (swap_pos_l == BATCHSIZE-1) {
            swap_pos_l=0, swap_start_l=0, swap_pos_b++;
            puts("side reach");
        } else
            swap_pos_l++, swap_pos_b += (b_change_needed ? 1 : 0);

        //update free_pos and, if necessary, curr_batch: (2/2)
        if ( ++(batches[curr_batch].free_pos) == BATCHSIZE )
            curr_batch++;
    }

    // Just info:
    puts("The following batches are not full (0-based index).");
    for (int b=0; b<n_batches; b++)
        if (batches[b].free_pos != BATCHSIZE) {
            if (!batches[b].free_pos) {
                printf("%d batches will be printed.\n", b);
                break;
            }
            printf("%3d) %d\n", b, batches[b].free_pos);
        }
    
    
    /*** WRITING AND FREEING ***/
    for (int b=0; b<n_batches; b++)
        for (int s=0; s<batches[b].free_pos; s++)
            fwrite(batches[b].lines[s], strlen(batches[b].lines[s]), 1, csv_out);

    for (int b=0; b<n_batches; b++)
        for (int s=0; s<batches[b].free_pos; s++)
            free(batches[b].lines[s]);
    
    fclose(csv);
    fclose(csv_out);
    return 0;
}