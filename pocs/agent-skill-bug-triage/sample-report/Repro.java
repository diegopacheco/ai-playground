package org.springframework.boot.loader.net.protocol.jar;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.lang.management.ManagementFactory;
import java.net.URL;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import com.sun.management.ThreadMXBean;

import org.springframework.boot.loader.net.protocol.Handlers;

public class Repro {

	private static final int WARMUP = 200_000;

	private static final int MEASURE = 200_000;

	public static void main(String[] args) throws Exception {
		File file;
		if (args.length > 0) {
			file = new File(args[0]);
		}
		else {
			file = File.createTempFile("bug-triage-51503", ".jar");
			file.deleteOnExit();
			writeOuterJar(file);
		}

		Handlers.register();

		URL missing = JarUrl.create(file, "nested.jar", "missing/does-not-exist.dat");
		URL present = JarUrl.create(file, "nested.jar", "3.dat");

		verify(present, missing);

		Optimizations.enable(true);
		JarUrlConnection.open(missing);

		for (int i = 0; i < WARMUP; i++) {
			JarUrlConnection.open(missing);
		}

		ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
		long id = Thread.currentThread().threadId();
		long before = threads.getThreadAllocatedBytes(id);
		long start = System.nanoTime();
		for (int i = 0; i < MEASURE; i++) {
			JarUrlConnection.open(missing);
		}
		long nanos = System.nanoTime() - start;
		long allocated = threads.getThreadAllocatedBytes(id) - before;

		System.out.printf("allocated %d B/op%n", allocated / MEASURE);
		System.out.printf("time %.1f ns/op%n", (double) nanos / MEASURE);
	}

	private static void verify(URL present, URL missing) throws Exception {
		if (JarUrlConnection.open(present).getJarEntry() == null) {
			throw new IllegalStateException("cold lookup lost 3.dat");
		}
		if (JarUrlConnection.open(present).getJarEntry() == null) {
			throw new IllegalStateException("warm lookup lost 3.dat");
		}
		try {
			JarUrlConnection.open(missing).getJarEntry();
			throw new IllegalStateException("missing entry was reported as found");
		}
		catch (FileNotFoundException ex) {
			System.out.println("verify ok: cold hit, warm hit, missing entry not found");
		}
	}

	private static void writeOuterJar(File file) throws Exception {
		byte[] nested = nestedJarBytes();
		try (ZipOutputStream out = new ZipOutputStream(new FileOutputStream(file))) {
			ZipEntry entry = new ZipEntry("nested.jar");
			entry.setMethod(ZipEntry.STORED);
			entry.setSize(nested.length);
			entry.setCompressedSize(nested.length);
			CRC32 crc = new CRC32();
			crc.update(nested);
			entry.setCrc(crc.getValue());
			out.putNextEntry(entry);
			out.write(nested);
			out.closeEntry();
		}
	}

	private static byte[] nestedJarBytes() throws Exception {
		ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		try (ZipOutputStream out = new ZipOutputStream(bytes)) {
			out.putNextEntry(new ZipEntry("3.dat"));
			out.write(new byte[] { 3 });
			out.closeEntry();
		}
		return bytes.toByteArray();
	}

}
