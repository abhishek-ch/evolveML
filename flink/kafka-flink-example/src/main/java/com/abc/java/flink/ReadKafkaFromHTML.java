package com.abc.java.flink;

import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import java.util.Properties;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer082;
import org.apache.flink.streaming.util.serialization.SimpleStringSchema;
import org.apache.flink.util.Collector;
import org.jsoup.Jsoup;

public class ReadKafkaFromHTML {

	public static void main(String[] args) throws Exception {
		final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
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

		messageStream.rebalance().map(new MapFunction<String, String>() {
			private static final long serialVersionUID = -4263083077217874531L;

			@Override
			public String map(String value) throws Exception {
				value = Jsoup.parse(value).text();
//				DataStreamSource<String> text = env.fromElements(Jsoup.parse(value).text());
//				// text.flatMap(new Tokenizer()).keyBy(0).sum(1)
//				SingleOutputStreamOperator<Tuple2<String, Integer>, ?> counts =
//				// split up the lines in pairs (2-tuples) containing: (word,1)
//				text.flatMap(new Tokenizer())
//						// group by the tuple field "0" and sum up tuple field
//						// "1"
//						.keyBy(0).sum(1);

				return ZonedDateTime.now().format(DateTimeFormatter.RFC_1123_DATE_TIME)+" : "+value.length();

			}
		}).print();
		
		
		env.execute();
	}

	// private static DataSet<String> getTextDataSet(ExecutionEnvironment env,
	// String textpath) {
	// // read the text file from given input path
	// return env.readTextFile(textPath);
	// }

	private static final class Tokenizer implements FlatMapFunction<String, Tuple2<String, Integer>> {

		/**
		 * 
		 */
		private static final long serialVersionUID = 7187013284840271463L;

		@Override
		public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
			// normalize and split the line
			String[] tokens = value.toLowerCase().split("\\W+");

			// emit the pairs
			for (String token : tokens) {
				if (token.length() > 0) {
					out.collect(new Tuple2<String, Integer>(token, 1));
				}
			}
		}
	}

}
