package com.abc.kafka.example;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.stream.Stream;

import kafka.javaapi.producer.Producer;
import kafka.producer.KeyedMessage;
import kafka.producer.ProducerConfig;

public class KafkaTextReaderProducer {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		
		Properties props = new Properties();
		props.put("metadata.broker.list", "localhost:9091");
		props.put("producer.type", "sync");
		props.put("serializer.class", "kafka.serializer.StringEncoder");
		//props.put("partitioner.class", "com.abc.kafka.example.KafkaPartitionar");
		props.put("request.required.acks", "1");
		ProducerConfig config = new ProducerConfig(props);
		Producer<String, String> producer = new Producer<String, String>(config);
		
		Files.walk(Paths.get("/Volumes/work/data/kaggle/dato/0/")).forEach(filePath -> {
			int lastIndexOf = filePath.toString().lastIndexOf(File.separator);
			String substring = filePath.toString().substring(lastIndexOf+1);
			//System.err.println("substringsubstring "+substring);
			if (Files.isRegularFile(filePath) && !substring.startsWith(".")) {
				System.out.println(filePath);
				
				//read each line of the file
				try (Stream<String> stream = Files.lines(filePath, Charset.defaultCharset())) {
					//stream.forEach(System.out::println);
					stream.forEach(val -> {
						KeyedMessage<String, String> data = new KeyedMessage<String, String>("test", val);
						System.err.println("datadata "+data);
						producer.send(data);
					});
					
				} catch (IOException ex) {
					// do something with exception
				}

			}
		});
	}

}
