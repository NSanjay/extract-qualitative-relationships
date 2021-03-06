package stanford_test.candidate_generation;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.tokensregex.PhraseTable.PhraseStringCollection;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
//import edu.stanford.nlp.pipeline.CoreD
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.Properties;
import java.util.stream.Collectors;
import org.json.JSONObject;


public class ParaPhrases {

    public static void main(String[] args) throws IOException {
    	int maxWordsInCandidate = 6;
        String dataset = "test";
        // creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // read some text in the text variable
        //String text = "What is the Weather in Bangalore right now?";
        String text = "Musicland is trying to embrace the Internet and emulate the atmosphere of retail chains like Starbucks.";
        // create an empty Annotation just with the given text
        String text1 = "A jetliner has high resistance moving through cloudy air and low resistance moving through thin air.";
        String text2 = "Jim wants to cross a river and has a one inch thick wood plank and a four inch thick wood plank. Which plank is more likely to break? He hates to cross though.";
        CoreDocument document = new CoreDocument(text2);
        //System.out.println(text2.replaceAll("\\.$",""));
        //System.exit(0);
        /*pipeline.annotate(document);
        List<CoreSentence> sentences = document.sentences();
        List<String> nouns = new ArrayList<>();
        for(CoreSentence sentence : sentences){
        	//sen.tokens().get(0)
        	for (CoreLabel label: sentence.tokens()) {
        		if (label.tag().startsWith("NN")) {
        			nouns.add(label.originalText());
        		}
        		System.out.println(label.originalText());
        		System.out.println(label.tag());
        	}
        }
        System.out.println(nouns);
        List<String> tokens = sentences.stream()
        		.map(sent -> sent.constituencyParse())
        		.map(n -> getAllPhrasalNodes(n))
        		.flatMap(Collection::stream)
        		.map( n -> removeArticles(getPhraseString(n).toLowerCase()))
        		.collect(Collectors.toList());
        System.out.println(tokens);*/
        // run all Annotators on this text
       /* pipeline.annotate(document);
        
        List<Tree> constituencies = sentences.stream()
        		.map(s -> s.constituencyParse())
        		.collect(Collectors.toList());
        List<Tree> nodes = constituencies.stream()
        		.map(n -> getAllPhrasalNodes(n))
        		.collect(Collectors.toList())
                .stream()
        		.flatMap(Collection::stream)
        		.filter(n -> hasNoPreposition(n))
                .collect(Collectors.toList());
        */
        
        
        //List<Tree> newNodes = nodes.stream()
        //		.filter(n -> hasNoPreposition(n))
        //		.collect(Collectors.toList());
        /*CoreSentence sentence = document.sentences().get(0);
        System.out.println(document.sentences());
        Tree constituencyParse = sentence.constituencyParse();
        List<Tree> nodes = getAllPhrasalNodes(constituencyParse);*/
        /*List<Tree> d2nodes = nodes.stream()
        		.filter(n -> n.depth()==2)
        		.collect(Collectors.toList());*/
        /*List<String> phraseStrings = nodes.stream()
        		.map(n -> removeArticles(getPhraseString(n)))
        		.collect(Collectors.toList());
        int largestString = phraseStrings
        		.stream()
        		.map(s -> wordsInString(s))
        		.max(Comparator.comparing(Integer::valueOf))
        		.get();
        System.out.println(largestString);
        
        System.out.println(phraseStrings);
        System.out.println("All required phrasal nodes are:");
        System.out.print(nodes.size());
        //System.out.println(d2nodes.get(0).getChildrenAsList().get(0).nodeString());
        for (String s: phraseStrings) {
        	System.out.println(s);
        	System.out.println(wordsInString(s));
        	
        }
        System.out.println("Example: constituency parse");
        for (Tree somet : nodes) {
        	System.out.println(somet);
        }*/
        
        //System.out.println(constituencyParse);
        System.out.println("here");
       // System.out.println(constituencyParse.children()[0].children()[0].children()[0].children()[0].children()[0].nodeString());
        //TreeExtension ext = new TreeExtension();
        //System.out.println();
        //System.out.println(ext.treestringer(constituencyParse, "neg"));
        List<String> records = new ArrayList<>();
        
        
        //// Sanjay - changes start here
        try (BufferedReader br = new BufferedReader(new FileReader("/home/sanjay/Projects/Quartz_Dataset/data/QuaRTz-dataset-v1/"+dataset+".jsonl"))) {
        	String line;
        	while ((line = br.readLine()) != null) {
        		records.add(line);
        	}
        }
        

        List<JSONObject> jsons = records.stream()
        		.map(r -> new JSONObject(r.toString()))
        		.collect(Collectors.toList());
        /*List<String> onlyquestions = jsons.stream()
        		.map(j -> Arrays.asList(j.getString("question").split("\\(")).get(0))
        		.collect(Collectors.toList());
        List<String> optionas = jsons.stream()
        		.map(j -> Arrays.asList(j.getString("question").split("\\(")).get(1))
        		.map(s -> s.substring(3))
        		.map(s -> removeoptionpunctuations(s.toLowerCase().trim()))
        		.collect(Collectors.toList());
        List<String> optionbs = jsons.stream()
        		.map(j -> Arrays.asList(j.getString("question").split("\\(")).get(2))
        		.map(s-> s.substring(3))
        		.map(s -> removeoptionpunctuations(s.toLowerCase().trim()))
        		.collect(Collectors.toList());
        */

        List<String> paras = jsons.stream()
                .map(j-> j.getString("para").replaceAll("\\.$",""))
        		//.map(j-> j.getString("para"))
                .collect(Collectors.toList());

        List<CoreDocument> paradocs = paras.stream()
                .map(s -> new CoreDocument(s))
                .collect(Collectors.toList());
        
        // List<CoreDocument> optionadocs = optionas.stream()
        //         .map(s -> new CoreDocument(s))
        //         .collect(Collectors.toList());
        
        // List<CoreDocument> optionbdocs = optionbs.stream()
        //         .map(s -> new CoreDocument(s))
        //         .collect(Collectors.toList());
        
        
        // for (CoreDocument someDoc: optionadocs) {
        // 	pipeline.annotate(someDoc);
        // }
        
        for(CoreDocument doc : paradocs) {
            pipeline.annotate(doc);
        }

        // for (CoreDocument someDoc: optionbdocs) {
        // 	pipeline.annotate(someDoc);
        // }
        
        // List<List<String>> llspara =  optionadocs.stream()
        //         .map(d -> d.sentences().get(0))
        //         .collect(Collectors.toList())
        //         .stream()
        //         .map(c -> getAllPhrasalNodes(c.constituencyParse()))
        //         .map(l -> modifylist(l))
        //         .collect(Collectors.toList());

        // List<List<String>> llsa =  optionadocs.stream()
        // 		.map(d -> d.sentences().get(0))
        // 		.collect(Collectors.toList())
        // 		.stream()
        // 		.map(c -> getAllPhrasalNodes(c.constituencyParse()))
        // 		.map(l -> modifylist(l))
        // 		.collect(Collectors.toList());
        		
        // List<List<String>> llsb =  optionbdocs.stream()
        // 		.map(d -> d.sentences().get(0))
        // 		.collect(Collectors.toList())
        // 		.stream()
        // 		.map(c -> getAllPhrasalNodes(c.constituencyParse()))
        // 		.map(l -> modifylist(l))
        // 		.collect(Collectors.toList());
        
         
        // List<CoreDocument> coredocs = onlyquestions.stream()
        //         .map(s -> new CoreDocument(s))
        //         .collect(Collectors.toList());
        // for (CoreDocument someDoc: coredocs) {
        // 	pipeline.annotate(someDoc);
        // }
        
        List<List<CoreSentence>> paratrees = paradocs.stream()
        		.map(d -> d.sentences())
        		.collect(Collectors.toList());
        System.out.println("here1");
        List<List<String>> phraseStringsPerTree = new ArrayList<>();
        int index = 0;
        for (List<CoreSentence> coresPerExample: paratrees) {
        	List<CoreLabel> nountokens = coresPerExample.stream()
        			.map(sent -> sent.tokens())
        			.flatMap(Collection::stream)
        			.collect(Collectors.toList());
        	List<String> nouns = nountokens.stream()
        			.filter(tok -> tok.tag().startsWith("NN"))
        			.map(tok -> tok.originalText().toLowerCase())
        			.collect(Collectors.toList());
        	List<Tree> ts = coresPerExample.stream()
        			.map(c -> getAllPhrasalNodes(c.constituencyParse()))
        			.collect(Collectors.toList())
        			.stream()
        			.flatMap(Collection::stream)
        			.collect(Collectors.toList());
        	
        	List<String> reqstrings = ts.stream()
        			//.map(t -> removeArticles(getPhraseString(t).toLowerCase()))
        			.map(t -> getPhraseString(t).toLowerCase())
//        			.filter(s -> wordsInString(s)<=maxWordsInCandidate)
        			.filter(s -> wordNotInList(s))
        			.filter(s -> s.indexOf("_")<0)
        			.collect(Collectors.toList());
        	//List<String> nonArticleStrings = reqstrings.stream()
        	//		.filter(s -> containsArticle(s))
        	//		.map(s-> removeArticles(s))
        	//		.collect(Collectors.toList());
        	//reqstrings.addAll(nonArticleStrings);
        	//reqstrings.add(removeArticles(optionas.get(index).toLowerCase()));
        	//reqstrings.add(removeArticles(optionbs.get(index).toLowerCase()));
        	//reqstrings.addAll(llsa.get(index));
        	//reqstrings.addAll(llsb.get(index));
        	reqstrings.addAll(nouns);
        	
        	reqstrings = reqstrings.stream()
        			.map(s-> removepunctuations(s.trim()))
        			.filter(s -> wordsInString(s)<=maxWordsInCandidate)
        			.filter(s -> wordNotInList(s))
        			.map(s -> s.replace("  ", " "))
        			.collect(Collectors.toList());
        	
        	
        	phraseStringsPerTree.add(new ArrayList<>(new HashSet<>(reqstrings)));
        	index = index + 1;
        	
        }
        
        List<Integer> maxSizes = phraseStringsPerTree.stream()
        		.map(l -> getMaxSizeforPhraseStrings(l))
        		.collect(Collectors.toList());
        
        List<Integer> sizes = phraseStringsPerTree.stream()
        		.map(l -> l.size())
        		.collect(Collectors.toList());
        
        System.out.println(phraseStringsPerTree.size());
        System.out.println(jsons.get(0));
        System.out.println(phraseStringsPerTree.get(0));
        createFileForjsonsAndStrings(phraseStringsPerTree, jsons, dataset, maxWordsInCandidate);
        
        
        /*for (JSONObject js: jsons) {
        	
        }*/
        
       /* int largestnpsize = Collections.max(maxSizes);
        int largeIndex = maxSizes.indexOf(largestnpsize);
        System.out.println("largest noun phrase in dev dataset is:" + largestnpsize + "and occurs in " + largeIndex);
        System.out.println(phraseStringsPerTree.get(largeIndex));
        
        System.out.println("total phrases per record:");
        System.out.println(sizes);*/
        
        
        /*File fout = new File("\"C:\\\\Masters Studies\\\\interests\\\\question answering\\\\quarel-dataset-v1-nov2018\\\\quarel-dataset-v1\\\\quarel-cand-dev.json\"");
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
        for (int i=0; i<jsons.size();i++) {
        	JSONObject js = jsons.get(i);
        	js.append("cadidates", phraseStrings.get(i));
        	bw.write(js.toString());
        	bw.newLine();
        }
        bw.close();
        */
        
        
        //System.out.println(questrees.size());
        //JSONObject obj = new JSONObject(records.get(0));
        //System.out.println(records.get(0));
        //String question = obj.getString("question");
        //List<String> toks = Arrays.asList(question.split("\\("));
        //System.out.println(toks.get(1));
        //List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

       /* for (CoreMap sentence : sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                // this is the POS tag of the token
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                // this is the NER label of the token
                String ne = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);

                System.out.println(String.format("Print: word: [%s] pos: [%s] ne: [%s]", word, pos, ne));
            }
        }*/
        /*
        PrintWriter out = null;
        out = new PrintWriter("C:\\Masters Studies\\PLAI\\Project\\Sarcasm\\devNew.res");
		
        try (BufferedReader br = new BufferedReader(new FileReader("C:\\Masters Studies\\PLAI\\Project\\Sarcasm\\dev.tsv"))) {
            String line;
            int i= 0;
            while ((line = br.readLine()) != null) {
               List<String> tokens = new LinkedList<String>(Arrays.asList(line.split("\\s+")));
               String sentiment = tokens.get(0);
               tokens.remove(0);
               String newSentence = tokens.stream().collect(Collectors.joining(" "));
               
               document = new CoreDocument(newSentence);
               pipeline.annotate(document);
               sentence = document.sentences().get(0);
               constituencyParse = sentence.constituencyParse();
               String modifiedout = ext.treestringer(constituencyParse, sentiment);
               out.println(modifiedout);
               i++;
               if (i%100==0) {
            	   System.out.println(i);
               }
            }
        } catch (IOException e) {
        	System.out.println(e);
        }
        out.close();*/
    }

	private static String removeoptionpunctuations(String s) {
		if (s.endsWith(".")) {
			return s.replace(".", "");
		} else {
			return s;
		}
	}
	
    private static boolean containsArticle(String s) {
    	if (s.startsWith("a")) {
    		return true;
    	} else if (s.startsWith("the")) {
    		return true;
    	}
		return s.contains(" a ") || s.contains(" the ") || s.contains(" an ");
	}

	private static boolean wordNotInList(String word) {
    	List<String> words = Arrays.asList("_____", "the","this",""," ");
		return !words.contains(word);
	}

	private static void createFileForjsonsAndStrings(List<List<String>> phraseStringsPerTree, List<JSONObject> jsons, String dataset, int maxWordsInCandidate) {
    	System.out.println("phrase strings size" + phraseStringsPerTree.size());
        System.out.println("jsons size " + jsons.size());
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("/home/sanjay/Projects/Quartz_Dataset/data/QuaRTz-dataset-v1/"+dataset+"_chunks.jsonl"))) {
        	for (int i=0; i<jsons.size();i++) {
            	JSONObject js = jsons.get(i);
            	JSONObject jsonObject = new JSONObject();
            	jsonObject.put("candidates", new ArrayList<>(new HashSet<>(phraseStringsPerTree.get(i))));
            	jsonObject.put("id",js.get("id"));
            	bw.write(jsonObject.toString());
            	//bw.write(js.toString());
            	bw.newLine();
            }
        } catch (Exception e) {
			System.out.println("Exception faced during io");
		}
    }
    
	private static List<Tree> getAllPhrasalNodes(Tree constituencyParse) {
		// TODO Auto-generated method stub
		List<Tree> phrasalNodes = new ArrayList<>();
		if ((constituencyParse!=null)) {
			if ((constituencyParse.nodeString().startsWith("NP")) || (constituencyParse.nodeString().startsWith("NNP") )
					|| (constituencyParse.nodeString().startsWith("JJ")) || (constituencyParse.nodeString().startsWith("VP"))
		|| (constituencyParse.nodeString().startsWith("ADVP")) || (constituencyParse.nodeString().equals("S"))
		|| (constituencyParse.nodeString().startsWith("X"))
		) {
				phrasalNodes.add(constituencyParse);
			} 
		}
		for (Tree node: constituencyParse.getChildrenAsList()) {
			phrasalNodes.addAll(getAllPhrasalNodes(node));
		}
		return phrasalNodes;
	}
	
	private static String getPhraseString(Tree phrasalNode) {
		if (phrasalNode.depth()==0) {
			return phrasalNode.nodeString();
		}  else {
			if (phrasalNode.nodeString().startsWith("PRP") && phrasalNode.isPreTerminal()) {
				return "";
			}
//			if (phrasalNode.nodeString().startsWith("JJR") && phrasalNode.isPreTerminal()) {
//				return "";
//			}
//			if (phrasalNode.nodeString().startsWith("JJS") && phrasalNode.isPreTerminal()) {
//				return "";
//			}
//			if (phrasalNode.nodeString().startsWith("RBR") && phrasalNode.isPreTerminal()) {
//				return "";
//			}
//			if (phrasalNode.nodeString().startsWith("RBS") && phrasalNode.isPreTerminal()) {
//				return "";
//			}
			String totalString = "";
			for (Tree node: phrasalNode.getChildrenAsList()) {
				totalString = totalString + " "+ getPhraseString(node);
			}
			return totalString.substring(1).toLowerCase();
		}
	}
	private static int wordsInString(String s) {
		String text = s.trim();
		int words = text.isEmpty() ? 0 : text.split("\\s+").length;
		return words;
	}
	
	private static List<String> getPhraseStringsforTree(Tree tree) {
		List<Tree> nodes = getAllPhrasalNodes(tree);
		List<String> phraseStrings = nodes.stream()
        		.map(n -> getPhraseString(n))
        		.collect(Collectors.toList());
		return phraseStrings;
	}
	
	private static int getMaxSizeforPhraseStrings(List<String> phraseStrings) {
		if (phraseStrings.size()==0) {
			return 0;
		}
		int largestString = phraseStrings
        		.stream()
        		.map(s -> wordsInString(s))
        		.max(Comparator.comparing(Integer::valueOf))
        		.get();
		return largestString;
	}
	
	private static boolean hasNoPreposition(Tree node) {
		if (node.nodeString().startsWith("NP")) {
			if (node.numChildren()>0) {
				boolean hasnoprep = true;
				for (Tree childNode : node.getChildrenAsList()) {
					hasnoprep= hasnoprep && hasNoPreposition(childNode);
				}
				return hasnoprep;
			} else {
				return true;
			}
		}
		/*else if (node.nodeString().startsWith("PRP")) {
			return false;
		}*/
		return true;
	}
	
	private static String removeArticles(String somestring) {
		somestring = somestring.replaceAll(" a ", " ").replaceAll(" an ", " ").replaceAll(" the ", " ");
		if (somestring.startsWith("a ")) {
			return somestring.substring(2);
		} else if(somestring.startsWith("an ")) {
			return somestring.substring(3);
		} else if (somestring.startsWith("the ")) {
			return somestring.substring(4);
		}
		return somestring;
	}
	
	private static String removepunctuations(String s) {
		s = s.replace(".", "");
		s = s.replace("?", "");
		return s;
	}
	
	private static List<String> modifylist(List<Tree> trees) {
		return trees.stream()
				.map(t -> getPhraseString(t))
				.collect(Collectors.toList());
	}
}