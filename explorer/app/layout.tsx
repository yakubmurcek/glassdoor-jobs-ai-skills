import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { MobileNav } from "@/components/layout/mobile-nav";
import { TooltipProvider } from "@/components/ui/tooltip";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Skills Explorer — Master's thesis companion",
  description:
    "Interactive explorer for the thesis on AI skill requirements in IT job postings (US vs Germany vs India).",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-svh antialiased`}
      >
        <NuqsAdapter>
          <TooltipProvider delayDuration={150}>
            <div className="flex min-h-svh">
              <Sidebar />
              <main className="flex min-h-svh flex-1 flex-col">
                <div className="flex-1 px-4 py-6 md:px-8 md:py-8">{children}</div>
                <MobileNav />
              </main>
            </div>
          </TooltipProvider>
        </NuqsAdapter>
      </body>
    </html>
  );
}
