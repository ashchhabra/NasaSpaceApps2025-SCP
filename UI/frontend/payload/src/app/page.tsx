import Image from "next/image";
import Head from "next/head";
import Entry from "../components/Entry";

export default function Home() {
  return (
    <>
      <div className="container">
       <Entry />       
      </div>
    </>
  );
}

export const metadata = {
  title: 'Payload',
  description: 'Hello! Hope you\'re having a great day!',
  icons: {
    icon: '/favicon.ico',
  },
};

