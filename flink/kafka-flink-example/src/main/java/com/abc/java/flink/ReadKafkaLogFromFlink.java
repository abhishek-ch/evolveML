package com.abc.java.flink;

import java.util.Properties;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer082;
import org.apache.flink.streaming.util.serialization.SimpleStringSchema;

public class ReadKafkaLogFromFlink {

	public static void main(String[] args) throws Exception {
		// create execution environment
		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		// parse user parameters
		// ParameterTool parameterTool = ParameterTool.fromArgs(args);

		Properties props = new Properties();
		props.put("bootstrap.servers", "localhost:9091");
		props.put("zookeeper.connect", "localhost:2181");
		props.put("group.id", "myGroup");

		// new FlinkKafkaConsumer082<>("topic", new
		// SimpleStringSchema(), properties)
		DataStream<String> messageStream = env
				.addSource(new FlinkKafkaConsumer082<>("test", new SimpleStringSchema(), props));

		// print() will write the contents of the stream to the TaskManager's
		// standard out stream
		// the rebelance call is causing a repartitioning of the data so that
		// all machines
		// see the messages (for example in cases when "num kafka partitions" <
		// "num flink operators"
		messageStream.rebalance().map(new MapFunction<String, String>() {
			private static final long serialVersionUID = -6867736771747690202L;

			@Override
			public String map(String value) throws Exception {
				return "Kafka and Flink says: " + value;
			}
		}).print();

		env.execute();
	}

}
