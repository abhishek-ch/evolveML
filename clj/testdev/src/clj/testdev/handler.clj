(ns testdev.handler
  (:require [compojure.core :refer [GET defroutes]]
            [compojure.route :refer [not-found resources]]
            [ring.middleware.defaults :refer [site-defaults wrap-defaults]]
            [hiccup.core :refer [html]]
            [hiccup.page :refer [include-js include-css]]
            [prone.middleware :refer [wrap-exceptions]]
            [ring.middleware.reload :refer [wrap-reload]]
            [testdev.view.contents :as contents]
            [environ.core :refer [env]]
            [selmer.parser :as parser]
            ))

(def home-page
  (html
   [:html
    [:head
     [:title "Debdoot Agency"]
     [:meta {:charset "utf-8"}]
     [:meta {:http-equiv "X-UA-Compatible" :content "IE=edge,chrome=1"}]
     [:meta {:name "viewport" :content "width=device-width, initial-scale=1, maximum-scale=1"}]
     (include-css "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css"
                  "css/styles.css")]
    [:body
     [:div#app
      [:h3 "ClojureScript has not been compiled!"]
      [:p "please run "
       [:b "lein figwheel"]
       " in order to start the compiler"]]
     (include-js "js/app.js")
     (include-js "https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"
                 "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"
                 "https://maps.googleapis.com/maps/api/js?key=AIzaSyBVe4qpNwgKVUU_g62OSqpXi6H6FDp5UyU"
                 )
     ;[:script {:src "https://maps.googleapis.com/maps/api/js?key=AIzaSyBVe4qpNwgKVUU_g62OSqpXi6H6FDp5UyU"}]
     [:script "google_maps.core.main();"]
     ]]))

;https://github.com/yokolet/hiccup-samples
;http://www.bootply.com/96188
;https://github.com/reagent-project/reagent-forms
;https://github.com/reagent-project/reagent-template - run application

(def headerLinks
  [{:name "Link1" :href "http://www.google.com" :class "glyphicon glyphicon-chevron-right"}
  {:name "Link2" :href "http://www.google.com" :class "glyphicon glyphicon-user"}
  {:name "Link3" :href "http://www.google.com" :class "glyphicon glyphicon-lock"}
  {:name "Link4" :href "http://www.google.com" :class "glyphicon glyphicon-cog"}]
  )

(comment (defn headerLinksDef

               (println "ok" (doseq [record headerLinks
                                     [k v] record]
                                    (println k v)) )
               ))



(defn resource [r]
  (-> (Thread/currentThread)
    (.getContextClassLoader)
    (.getResource r)
    slurp))


(defn application [title]
  (html
    [:head [:meta {:charset "utf-8"}]
     [:meta {:http-equiv "X-UA-Compatible" :content "IE=edge,chrome=1"}]
     [:meta {:name "viewport" :content "width=device-width, initial-scale=1, maximum-scale=1"}]
     [:title title]
     ]
  ))


(defn not-founds []
  [:div
   [:h1 {:class "info-worning"} "Page Not Found"]
   [:p "There's no requested page. "]
   (:button.btn.btn-default {:class "btn btn-primary"} "/" "Take me to Home")])

(defroutes routes
  (GET "/" [] home-page)
  (GET "/about" [] (application "signup"
                     {:forms-css (resource "reagent-forms.css")
                      :json-css (resource "json.human.css")}))
  (resources "/")
  (not-found "Not Found"))


(def app
  (let [handler (wrap-defaults #'routes site-defaults)]
    (if (env :dev) (-> handler wrap-exceptions wrap-reload) handler)))


