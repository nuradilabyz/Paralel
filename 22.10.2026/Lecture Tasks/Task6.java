import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
public class Task6 {
    static volatile int counter;
    static final int WORKERS=4,ITERATIONS=2_000_000,EXPECTED=WORKERS*ITERATIONS;
    static double run(boolean atomic,int iterations) throws Exception {
        counter=0;AtomicInteger safe=new AtomicInteger();
        CountDownLatch ready=new CountDownLatch(WORKERS),go=new CountDownLatch(1);
        Thread[] threads=new Thread[WORKERS];
        for(int w=0;w<WORKERS;w++) {
            threads[w]=new Thread(()->{
                ready.countDown();try{go.await();}catch(InterruptedException e){throw new RuntimeException(e);}
                if(atomic)for(int i=0;i<iterations;i++)safe.incrementAndGet();
                else for(int i=0;i<iterations;i++)counter++;
            });threads[w].start();
        }
        ready.await();long s=System.nanoTime();go.countDown();for(Thread t:threads)t.join();double elapsed=Bench.ms(s);
        int actual=atomic?safe.get():counter,expected=WORKERS*iterations;
        if(atomic)Bench.check(actual==expected,"Atomic lost updates");
        if(iterations==ITERATIONS)System.out.printf("%s | Expected %d | Actual %d | Lost %.6f%% | Time %.6f ms%n",atomic?"AtomicInteger":"Racy volatile ++",expected,actual,100.0*(expected-actual)/expected,elapsed);
        return elapsed;
    }
    public static void main(String[] args)throws Exception {
        Bench.header("TASK 6 - Synchronization and data loss");
        System.out.println("Volatile provides visibility but ++ remains a non-atomic read-modify-write. Correct version uses hardware-backed AtomicInteger.");
        run(false,100_000);run(true,100_000);
        double[] racy=new double[3],safe=new double[3];
        for(int r=0;r<3;r++){
            if(r%2==0){racy[r]=run(false,ITERATIONS);safe[r]=run(true,ITERATIONS);}
            else{safe[r]=run(true,ITERATIONS);racy[r]=run(false,ITERATIONS);}
        }
        java.util.Arrays.sort(racy);java.util.Arrays.sort(safe);
        System.out.printf("Median atomic synchronization penalty: %+.6f ms (%+.3f%%); synchronized / racy time: %.3fx%n",safe[1]-racy[1],100*(safe[1]/racy[1]-1),safe[1]/racy[1]);
        System.out.println("Atomic operations can outperform contended volatile ++; a positive time penalty is not guaranteed. Race results vary by run.");
    }
}
