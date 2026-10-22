import java.lang.invoke.*;
import java.util.concurrent.*;
public class Task3 {
    static final VarHandle ELEMENT=MethodHandles.arrayElementVarHandle(long[].class);
    static volatile long STEP=1;
    static final int ITERATIONS=50_000_000;
    static double run(int workers,int iterations,boolean local) throws Exception {
        long[] counts=new long[workers];
        CountDownLatch ready=new CountDownLatch(workers),go=new CountDownLatch(1);
        Thread[] threads=new Thread[workers];
        for(int w=0;w<workers;w++) { final int k=w;
            threads[w]=new Thread(()->{
                ready.countDown();
                try { go.await(); } catch(InterruptedException e) { throw new RuntimeException(e); }
                if(local) {
                    long value=0;
                    for(int j=0;j<iterations;j++) value+=STEP;
                    ELEMENT.setOpaque(counts,k,value);
                } else {
                    for(int j=0;j<iterations;j++) ELEMENT.setOpaque(counts,k,(long)ELEMENT.getOpaque(counts,k)+STEP);
                }
            }); threads[w].start();
        }
        ready.await(); long s=System.nanoTime(); go.countDown();
        for(Thread t:threads)t.join(); double elapsed=Bench.ms(s);
        for(long v:counts) Bench.check(v==iterations,"Counter mismatch");
        return elapsed;
    }
    public static void main(String[] args) throws Exception {
        Bench.header("TASK 3 - False sharing profile");
        int physical;
        if(System.getProperty("os.name").contains("Mac")) {
            Process proc=new ProcessBuilder("sysctl","-n","hw.physicalcpu").start();
            physical=Integer.parseInt(new String(proc.getInputStream().readAllBytes()).trim()); proc.waitFor();
        } else {
            physical=Integer.getInteger("physical.cores",0);
            if(physical<1) throw new IllegalArgumentException("Specify physical core count with -Dphysical.cores=N");
        }
        System.out.printf("Physical cores / workers: %d; increments per thread: %d%n",physical,ITERATIONS);
        System.out.println("Adjacent long slots; opaque accesses prevent JIT from removing shared updates. STEP volatile read is common to both loops.");
        System.out.println("Local version uses a private accumulator and publishes once; no padding assumed. This CPU may use cache lines larger than 64 bytes.");
        run(physical,100_000,false);run(physical,100_000,true);
        double[] shared=new double[3],local=new double[3];
        for(int r=0;r<3;r++) {
            if(r%2==0) {shared[r]=run(physical,ITERATIONS,false);local[r]=run(physical,ITERATIONS,true);}
            else {local[r]=run(physical,ITERATIONS,true);shared[r]=run(physical,ITERATIONS,false);}
        }
        java.util.Arrays.sort(shared);java.util.Arrays.sort(local);
        System.out.printf("Unpadded Shared: %.6f ms%nLocal: %.6f ms%nShared / Local ratio: %.3fx%n",shared[1],local[1],shared[1]/local[1]);
        System.out.println("Median of 3; release-to-join timing. Ratio includes repeated shared-store cost; hardware counters would be needed to isolate invalidations.");
    }
}
