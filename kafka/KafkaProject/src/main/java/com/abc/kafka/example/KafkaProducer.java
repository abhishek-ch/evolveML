package com.abc.kafka.example;

import java.util.Date;
import java.util.Properties;
import java.util.Random;

import kafka.javaapi.producer.Producer;
import kafka.producer.KeyedMessage;
import kafka.producer.ProducerConfig;

public class KafkaProducer {

	public static void main(String[] args) {
		long events = 100;
		Random rnd = new Random();

		Properties props = new Properties();
		props.put("metadata.broker.list", "localhost:9091");
		props.put("producer.type", "sync");
		props.put("serializer.class", "kafka.serializer.StringEncoder");
		//props.put("partitioner.class", "com.abc.kafka.example.KafkaPartitionar");
		props.put("request.required.acks", "1");
		ProducerConfig config = new ProducerConfig(props);
		Producer<String, String> producer = new Producer<String, String>(config);
		for (long nEvents = 0; nEvents < events; nEvents++) {
			System.out.println("creating event " + nEvents);
			long runtime = new Date().getTime();
			String ip = "Abhishek_Test" + rnd.nextInt(255);
			String msg = runtime + ",www.abc.com," + ip;
			KeyedMessage<String, String> data = new KeyedMessage<String, String>("test", ip, msg);
			producer.send(data);
		}
		producer.close();
	}

}
