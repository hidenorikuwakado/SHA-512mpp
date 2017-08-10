/* Time-stamp: <main.c 2017-05-18 09:14:47 Hidenori Kuwakado> */
/*
 * An AGGH matrix that is used in a message preprocessing is
 * generated. Elements of the AGGH matrix are values that are
 * related to pi (circular constant).
 */

#include <assert.h>
#include <ctype.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


/* ------------------------------------------------------------ */

#define PROGRAM_NAME "matrixGenerator"
#define PROGRAM_VERSION "1.0"
#define PROGRAM_COPYRIGHT "Copyright (C) 2017 Hidenori Kuwakado"

/* For debug code */
#define DO 1
#define xDO 0

/* Maximun length of an array name */
#define MaxLengthOfNameString (256+1)

/* This macro needs assert.h. */
#define MustBe(expr) \
  ((expr) \
   ? __ASSERT_VOID_CAST(0) \
   : __assert_fail(__STRING(expr), __FILE__, __LINE__, __ASSERT_FUNCTION))

/* The number of elements in an array */
#define NELMS(a) (sizeof(a)/sizeof(a[0]))

/* Print Bealen */
#define ToLogicStr(a) ((a) ? "true" : "false")


/* ------------------------------------------------------------ */

/* Command-line options */
typedef struct  {
    /* Help message */
    bool help;
    /* Output of a generated matrix */
    FILE *matrixFile;
    /*elements increases monotone. */
    bool mono;
    /* The sum of all the row elements increases monotone. */
    bool monoAddAll;
    /* Array name */
    char name[MaxLengthOfNameString];
    /* The number of rows/columns of a generated matrix */
    int numRows;
    int numColumns;
    /* The version of this source code */
    bool version;
    /* Generating the all-zero matrix */
    bool zero;
} option_t;


static void usage(void);
static void printCommandLineOptions(FILE *fp, const option_t *opt);
static uint8_t *computePi(const size_t rowColumn);


/* ------------------------------------------------------------ */

int
main(int argc, char *argv[])
{
    option_t opt = {
        .help = false,
        .matrixFile = stdout,
        .mono = false,
        .monoAddAll = false,
        .name = "array",
        .numRows = 8,
        .numColumns = 8,
        .version = false,
        .zero = false,
    };

    while (1) {
        /* The following is used for lables of case statements. The
         * maximum number of items is 64 because '?' that is a return
         * value is 64 in decimal and the first item is regarded as 0
         * in decimal. */
        enum {
            Help,
            MatrixFile,
            Mono,
            MonoAddAll,
            Name,
            NumRows,
            NumColumns,
            Version,
            Zero,
        };
        const struct option longopts[] = {
            { "help", no_argument, NULL, Help},
            { "matrixFile", required_argument, NULL, MatrixFile},
            { "mono", no_argument, NULL, Mono},
            { "monoAddAll", no_argument, NULL, MonoAddAll},
            { "name", required_argument, NULL, Name},
            { "numRows", required_argument, NULL, NumRows},
            { "numColumns", required_argument, NULL, NumColumns},
            { "version", no_argument, NULL, Version},
            { "zero", no_argument, NULL, Zero},
            { NULL, 0, NULL, 0 },
        };
        int longindex = 0;
        int c = getopt_long_only(argc, argv, "", longopts, &longindex);
        if (c == -1) { break; }
        switch (c) {
        case Help:
            usage();
            exit(EXIT_SUCCESS);
            break;
        case MatrixFile:
            opt.matrixFile = fopen(optarg, "w");
            MustBe(opt.matrixFile != NULL);
            break;
        case Mono:
            opt.mono = true;
            break;
        case MonoAddAll:
            opt.monoAddAll  = true;
            break;
        case Name:
            strncpy(opt.name, optarg, sizeof(opt.name)-1);
            opt.name[sizeof(opt.name)-1] = '\0';
            break;
        case NumRows:
            opt.numRows = atoi(optarg);
            if (opt.numRows <= 0) {
                fprintf(stderr, "--numRows require a positive integer. %d was given.\n", opt.numRows);
                exit(EXIT_FAILURE);
            }
            break;
        case NumColumns:
            opt.numColumns = atoi(optarg);
            if (opt.numColumns <= 0) {
                fprintf(stderr, "--numColumns require a positive integer. %d was given.\n", opt.numColumns);
                exit(EXIT_FAILURE);
            }
            break;
        case Version:
            printf("%s %s\n", PROGRAM_NAME, PROGRAM_VERSION);
            printf("%s\n", PROGRAM_COPYRIGHT);
            exit(EXIT_SUCCESS);
            break;
        case Zero:
            opt.zero = true;
            break;
        case '?':
            /* fall through */
        default:
            /* The given (unrecognized) option is displayed by getopt_long_only(). */
            usage();
            exit(EXIT_FAILURE);
        }
    }
    if (xDO) {
        printCommandLineOptions(stdout, &opt);
    }


    /* Matrix with 8-bit elements */
    uint8_t *a = (uint8_t *)malloc(opt.numRows * opt.numColumns);
    MustBe(a != NULL);

    if (opt.zero == true) {
        /* All-zero matrix */
        memset(a, 0x00, opt.numRows * opt.numColumns);
    } else if (opt.mono == true) {
        for (int i = 0; i < opt.numRows * opt.numColumns; ++i) {
            a[i] = (uint8_t)(i & 0xff);
        }
    } else if (opt.monoAddAll == true) {
        assert(opt.numRows % 4 == 0);
        assert(opt.numColumns % 2 == 0);

        for (int r = 0; r < opt.numRows; ++r) {
            uint8_t sum = 0;
            /* First, determine all the elements except for the last
             * element randomly. */
            for (int c = 0; c < opt.numColumns - 1; ++c) {
                a[opt.numColumns * r + c] = (uint8_t)(rand() & 0xff);
                sum += a[opt.numColumns * r + c];
            }
            /* Next, adjust the last element in such a way that the
             * sum becomes an expected value .*/
            a[opt.numColumns * r + opt.numColumns - 1] = r - sum;
        }
    } else {
        /* Use the circular constant as elements of a random matrix. */
        uint8_t *pi = computePi(opt.numRows * opt.numColumns);
        MustBe(pi != NULL);
        memcpy(a, pi, opt.numRows * opt.numColumns);
        free(pi);
    }

    /* Print the AGGH matrix in the convenient form. */
    srand(getpid());
    char matrixFileID[32] = { 'M' };
    sprintf(matrixFileID + 1, "%08x%08x", rand(), rand());
    fprintf(opt.matrixFile, "/*\n");
    printCommandLineOptions(opt.matrixFile, &opt);
    fprintf(opt.matrixFile, "*/\n");
    fprintf(opt.matrixFile, "#ifndef %s\n", matrixFileID);
    fprintf(opt.matrixFile, "#define %s\n", matrixFileID);
    fprintf(opt.matrixFile, "#include <stdint.h>\n");
    fprintf(opt.matrixFile, "static const uint8_t %s[%d * %d] = {\n",
            opt.name, opt.numRows, opt.numColumns);
    /* Column-oriented */
    for (int column = 0; column < opt.numColumns; ++column) {
        for (int row = 0; row < opt.numRows; ++row) {
            fprintf(opt.matrixFile, "0x%02x,", a[row * opt.numColumns + column]);
            if (row == opt.numRows - 1) {
                fputc('\n', opt.matrixFile);
            }
        }
    }
    fputs("};\n", opt.matrixFile);
    fprintf(opt.matrixFile, "#endif /* %s */\n", matrixFileID);

    /* Clean up the mess. */
    free(a);
    if (opt.matrixFile != stdout) {
        fclose(opt.matrixFile);
    }

    return EXIT_SUCCESS;
}


/* ------------------------------------------------------------ */

static void
usage(void)
{
    printf("Usage: %s [long option]\n", PROGRAM_NAME);
    puts("--help   Show this message.");
    puts("--matrixFile file   Output the matrix to the file (default: stdout).");
    puts("--mono   Use a montone incease as elements of the matrix.");
    puts("--monoAddAll   The sum of all the row elements is equal to the row index.");
    puts("--name array   Array name");
    puts("--numRows n   The number of rows");
    puts("--numColumns n   The number of columnss");    
    puts("--version   Show the version of this program.");
    puts("--zero   All-zero matrix");
}


static void
printCommandLineOptions(FILE *fp, const option_t *opt)
{
    fputs("Command-line options\n", fp);
    fprintf(fp, "--mono: %s\n", ToLogicStr(opt->mono));
    fprintf(fp, "--monoAddAll: %s\n", ToLogicStr(opt->monoAddAll));
    fprintf(fp, "--name: %s \n", opt->name);
    fprintf(fp, "--numRows: %d\n", opt->numRows);
    fprintf(fp, "--numColumns: %d\n", opt->numColumns);
    fprintf(fp, "--zero: %s\n", ToLogicStr(opt->zero));
}


/* ------------------------------------------------------------ */

/*
 * Compute pi (circular constant).  This code is given in
 * http://h2np.net/pi/index.html.
 */

/*
 * The Gauss AGM algorithm using Schonhage variation.
 * [1] Jorg Arndt, Christoph Haenel, "Pi Unleashed",
 *  pp. 93, Springer, 2001.
 * (C) 2003 Hironobu SUZUKI, licensed by GPL2
 *
 * Modified by Hidenori Kuwakado
 */
#include <stdio.h>
#include <gmp.h>
#include <time.h>

#define LOG_TEN_TWO  3.32192809488736234789
#define bprec(n) (int)(((n+10)*LOG_TEN_TWO)+2)

#if 0
/* Original code */
int main(int ac, char *av[])
#else
static uint8_t *
computePi(const size_t byteLength)
#endif
{
    if (xDO) {
        puts(__func__);
    }
    long int k, loopmax;
    mpf_t A, B, a, b, s, s_1, t, t1, t2, t3, c2;
#if 0
    /* Original code */
    long int prec, dprec;
    dprec = 1000000L; /* decimal precision */
    prec = bprec(dprec);  /* binary precision (plus alpha) */
    mpf_set_default_prec(prec);
    loopmax = 21;
#else
    /* The binary precision is bit length plus alpha (i.e., +64). */
    long int prec = byteLength * 8 + 64;
    mpf_set_default_prec(prec);
    loopmax = 20;
#endif

    mpf_init(A);  /* big A */
    mpf_init(B);  /* big B */
    mpf_init(a);  /* a */
    mpf_init(b);  /* b */
    mpf_init(s);  /* s(n) */
    mpf_init(s_1);  /* s(n-1) */
    mpf_init(t);  /* temporary */
    mpf_init(t1); /* temporary */
    mpf_init(t2); /* temporary */
    mpf_init(t3); /* temporary */
    mpf_init(c2); /* 2 constant */
    mpf_set_ui(A, 1);
    mpf_set_ui(a, 1);
    mpf_set_ui(t1, 1);
    mpf_div_ui(B, t1, 2);
    mpf_div_ui(s, t1, 2);
    mpf_set_ui(t1, 10);
    mpf_set_ui(c2, 2);
    for (k = 1; k <= loopmax; k++) {
        mpf_add(t1, A, B);  /* (A+B) */
        mpf_div_ui(t, t1, 4); /*  t = (A+B)/4 */
        mpf_sqrt(b, B); /* b = sqrt(B) */
        mpf_add(t1, a, b);  /* (a+b) */
        mpf_div_ui(a, t1, 2); /* a = (a+b)/2 */
        mpf_mul(A, a, a); /* A = a * a */
        mpf_sub(t1, A, t);  /*  (A-t) */
        mpf_mul_ui(B, t1, 2); /* B = (A - t) * 2 */
        mpf_sub(t1, B, A);  /* (B-A) */
        mpf_pow_ui(t2, c2, k);  /* 2^k */
        mpf_mul(t3, t1, t2);  /* (B-A) * 2^k  */
        mpf_set(s_1, s);
        mpf_add(s, s_1, t3);  /* s = s + (B-A) * 2^k */
    }
    mpf_add(t1, a, b);
    mpf_pow_ui(t2, t1, 2);
    mpf_mul_ui(t1, s, 2);
    mpf_div(A, t2, t1);
#if 0
    /* Original code */
    mpf_out_str(stdout, 10, dprec + 10, A);
    exit(0);
#else
    /* Store A to a temporary file, as a string of prec
     * bits. mpf_out_str() returns the number of bytes written, or if an
     * error occurred, return 0.
     */
    FILE *fp = tmpfile();
    MustBe(fp != NULL);
    size_t rv = mpf_out_str(fp, 2, prec, A);
    MustBe(rv != 0);
    if (xDO) {
        mpf_out_str(stdout, 2, prec, A);
        putchar('\n');
    }
    /* Since the string is 0.11001..., skip two letters (i.e.,
     * "0.").
     */
    fseek(fp, 2L, SEEK_SET);
    /* Get the mantissa of pi. */
    uint8_t *mantissa = (uint8_t *)malloc(byteLength);
    MustBe(mantissa != NULL);
    char buf[8 + 1];
    for (int i = 0; i < byteLength; ++i) {
        /* The string ends with "e2" (i.e., an exponent value). */
        if (fgets(buf, NELMS(buf), fp) != NULL &&
            strchr(buf, 'e') == NULL &&
            strlen(buf) == 8) {
            if (xDO) {
                printf("buf: %s\n", buf);
            }
            *(mantissa + i) = (uint8_t)strtol(buf, (char **)NULL, 2);
        } else {
            fputs("Not enough bits of pi\n", stderr);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);
    return mantissa;
#endif
}

/* end of file */
